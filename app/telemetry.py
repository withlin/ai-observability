import json
import time
from contextlib import contextmanager
from typing import Any

from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider, UpDownCounter
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    InMemoryMetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode

from config import Settings
from context import RunContext
from models import TokenUsage

_ROLE = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
TOKEN_BUCKETS = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]


def json_str(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def messages_json(role: str, text: str) -> str:
    return json_str([{"role": role, "parts": [{"type": "text", "content": text}]}])


def serialize_messages(messages) -> str:
    out = []
    for message in messages:
        role = _ROLE.get(getattr(message, "type", ""), getattr(message, "type", "user"))
        content = message.content
        text = content if isinstance(content, str) else json_str(content)
        parts = []
        if text:
            parts.append({"type": "text", "content": text})
        for tool_call in getattr(message, "tool_calls", None) or []:
            parts.append({
                "type": "tool_call",
                "id": tool_call.get("id"),
                "name": tool_call.get("name"),
                "arguments": tool_call.get("args", {}),
            })
        if role == "tool":
            parts.append({
                "type": "tool_call_response",
                "id": getattr(message, "tool_call_id", ""),
                "response": text,
            })
        out.append({"role": role, "parts": parts})
    return json_str(out)


class SpanHandle:
    def __init__(self, span: Span, capture_content: bool):
        self._span = span
        self._capture = capture_content

    @property
    def trace_id(self) -> int:
        return self._span.get_span_context().trace_id

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)

    def set_output(self, messages) -> None:
        if self._capture:
            self._span.set_attribute("gen_ai.output.messages", serialize_messages(messages))

    def set_result(self, result: Any) -> None:
        if self._capture:
            value = result if isinstance(result, str) else json_str(result)
            self._span.set_attribute("gen_ai.tool.call.result", value)

    def set_usage(self, usage: TokenUsage) -> None:
        self._span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
        self._span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
        if usage.cache_read_tokens:
            self._span.set_attribute("gen_ai.usage.cache_read.input_tokens", usage.cache_read_tokens)
        if usage.cache_write_tokens:
            self._span.set_attribute("gen_ai.usage.cache_write.input_tokens", usage.cache_write_tokens)
        if usage.reasoning_tokens:
            self._span.set_attribute("gen_ai.usage.reasoning.output_tokens", usage.reasoning_tokens)

    def set_error(self, exc: Exception) -> None:
        self._span.set_attribute("error.type", type(exc).__name__)
        self._span.set_status(Status(StatusCode.ERROR, str(exc)))
        self._span.record_exception(exc)


class Telemetry:
    """Facade over OpenTelemetry GenAI spans and metrics.

    The class owns its providers; it never mutates OTel global state. This keeps
    tests isolated and lets the application inject one instance explicitly.
    """

    def __init__(self, settings: Settings, *, in_memory: bool | None = None):
        self._settings = settings
        self.capture_content = settings.capture_content
        self._in_memory = settings.in_memory_telemetry if in_memory is None else in_memory

        resource = Resource.create({
            "service.name": "langgraph-agent",
            "service.version": "4.0.0",
            "deployment.environment": "local",
        })

        self.tracer_provider = TracerProvider(resource=resource)
        self.span_exporter: InMemorySpanExporter | None = None
        if self._in_memory:
            self.span_exporter = InMemorySpanExporter()
            self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.span_exporter))
        else:
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True)
                )
            )
        self.tracer = self.tracer_provider.get_tracer("gen_ai.langgraph")

        if self._in_memory:
            self.metric_reader = InMemoryMetricReader()
        else:
            self.metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=settings.otlp_endpoint,
                    insecure=True,
                    preferred_temporality={
                        Counter: AggregationTemporality.CUMULATIVE,
                        Histogram: AggregationTemporality.CUMULATIVE,
                        UpDownCounter: AggregationTemporality.CUMULATIVE,
                    },
                ),
                export_interval_millis=5000,
            )
        self.meter_provider = MeterProvider(
            resource=resource, metric_readers=[self.metric_reader], views=self._views()
        )
        self.meter = self.meter_provider.get_meter("gen_ai.langgraph")
        self._create_instruments()

    @staticmethod
    def _views() -> list[View]:
        return [
            View(instrument_type=Histogram, instrument_name="gen_ai.execute_tool.duration",
                 aggregation=ExplicitBucketHistogramAggregation(
                     [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])),
            View(instrument_type=Histogram,
                 instrument_name="gen_ai.client.inference.time_to_first_chunk",
                 aggregation=ExplicitBucketHistogramAggregation(
                     [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30])),
            View(instrument_type=Histogram,
                 instrument_name="gen_ai.client.inference.time_per_output_chunk",
                 aggregation=ExplicitBucketHistogramAggregation(
                     [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5])),
            View(instrument_type=Histogram, instrument_name="gen_ai.client.inference.duration",
                 aggregation=ExplicitBucketHistogramAggregation(
                     [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60])),
            View(instrument_type=Histogram, instrument_name="gen_ai.client.operation.duration",
                 aggregation=ExplicitBucketHistogramAggregation(
                     [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])),
            View(instrument_type=Histogram, instrument_name="gen_ai.invoke_agent.duration",
                 aggregation=ExplicitBucketHistogramAggregation(
                     [0.5, 1, 2, 5, 10, 20, 30, 60, 120])),
            View(instrument_type=Histogram, instrument_name="gen_ai.invoke_workflow.duration",
                 aggregation=ExplicitBucketHistogramAggregation(
                     [1, 2, 5, 10, 20, 30, 60, 120, 300])),
            View(instrument_type=Histogram, instrument_name="gen_ai.invoke_agent.inference_calls",
                 aggregation=ExplicitBucketHistogramAggregation([1, 2, 3, 4, 5, 6, 8, 10])),
            View(instrument_type=Histogram, instrument_name="gen_ai.invoke_agent.tool_calls",
                 aggregation=ExplicitBucketHistogramAggregation([1, 2, 3, 4, 5, 6, 8, 10])),
            View(instrument_type=Histogram,
                 instrument_name="gen_ai.client.inference.operation.input_tokens",
                 aggregation=ExplicitBucketHistogramAggregation(TOKEN_BUCKETS)),
            View(instrument_type=Histogram,
                 instrument_name="gen_ai.client.inference.operation.output_tokens",
                 aggregation=ExplicitBucketHistogramAggregation(TOKEN_BUCKETS)),
        ]

    def _create_instruments(self) -> None:
        meter = self.meter
        # Descriptions must match across every producer of a metric name.
        self.inference_duration = meter.create_histogram(
            "gen_ai.client.inference.duration", unit="s",
            description="GenAI inference operation duration")
        self.inference_ttfc = meter.create_histogram(
            "gen_ai.client.inference.time_to_first_chunk", unit="s",
            description="Time to first streamed chunk")
        self.inference_tpoc = meter.create_histogram(
            "gen_ai.client.inference.time_per_output_chunk", unit="s",
            description="Time per output chunk")
        self.inference_input_tokens = meter.create_counter(
            "gen_ai.client.inference.usage.input_tokens", unit="{token}",
            description="Number of input tokens used")
        self.inference_output_tokens = meter.create_counter(
            "gen_ai.client.inference.usage.output_tokens", unit="{token}",
            description="Number of output tokens used")
        self.inference_cache_read = meter.create_counter(
            "gen_ai.client.inference.usage.cache_read.input_tokens", unit="{token}",
            description="Number of cached input tokens read")
        self.inference_cache_write = meter.create_counter(
            "gen_ai.client.inference.usage.cache_write.input_tokens", unit="{token}",
            description="Number of input tokens written to cache")
        self.inference_reasoning = meter.create_counter(
            "gen_ai.client.inference.usage.reasoning.output_tokens", unit="{token}",
            description="Number of reasoning output tokens used")
        self.inference_op_input = meter.create_histogram(
            "gen_ai.client.inference.operation.input_tokens", unit="{token}",
            description="Input tokens per inference operation")
        self.inference_op_output = meter.create_histogram(
            "gen_ai.client.inference.operation.output_tokens", unit="{token}",
            description="Output tokens per inference operation")
        self.operation_duration = meter.create_histogram(
            "gen_ai.client.operation.duration", unit="s",
            description="GenAI operation duration")
        self.workflow_duration = meter.create_histogram(
            "gen_ai.invoke_workflow.duration", unit="s",
            description="GenAI workflow duration")
        self.agent_duration = meter.create_histogram(
            "gen_ai.invoke_agent.duration", unit="s",
            description="GenAI agent invocation duration")
        self.agent_inference_calls = meter.create_histogram(
            "gen_ai.invoke_agent.inference_calls", unit="{inference_call}",
            description="Inference calls per agent invocation")
        self.agent_tool_calls = meter.create_histogram(
            "gen_ai.invoke_agent.tool_calls", unit="{tool_call}",
            description="Tool calls per agent invocation")
        self.tool_duration = meter.create_histogram(
            "gen_ai.execute_tool.duration", unit="s",
            description="GenAI tool execution duration")

    def _inference_metric_attrs(self, model: str, response_model: str,
                                agent_name: str | None) -> dict[str, Any]:
        attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": self._settings.provider,
            "gen_ai.request.model": model,
            "gen_ai.response.model": response_model or model,
            "server.address": self._settings.server_address,
            "server.port": self._settings.server_port,
        }
        if agent_name:
            attrs["gen_ai.agent.name"] = agent_name
        return attrs

    @contextmanager
    def workflow_span(self, workflow_name: str, context: RunContext, input_text: str):
        attrs = {
            "gen_ai.operation.name": "invoke_workflow",
            "gen_ai.workflow.name": workflow_name,
            "gen_ai.conversation.id": context.conversation_id,
            "enduser.id": context.user_id,
        }
        with self.tracer.start_as_current_span(
            f"invoke_workflow {workflow_name}", kind=SpanKind.INTERNAL, attributes=attrs
        ) as span:
            handle = SpanHandle(span, self.capture_content)
            if self.capture_content:
                span.set_attribute("gen_ai.input.messages", messages_json("user", input_text))
            yield handle

    @contextmanager
    def agent_span(self, agent_name: str, description: str, model: str,
                   context: RunContext, system_prompt: str, tool_definitions: str | None,
                   messages):
        attrs = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": agent_name,
            "gen_ai.agent.description": description,
            "gen_ai.request.model": model,
            "gen_ai.conversation.id": context.conversation_id,
            "enduser.id": context.user_id,
        }
        with self.tracer.start_as_current_span(
            f"invoke_agent {agent_name}", kind=SpanKind.INTERNAL, attributes=attrs
        ) as span:
            handle = SpanHandle(span, self.capture_content)
            if self.capture_content:
                span.set_attribute("gen_ai.system_instructions",
                                   messages_json("system", system_prompt))
                if tool_definitions:
                    span.set_attribute("gen_ai.tool.definitions", tool_definitions)
                span.set_attribute("gen_ai.input.messages", serialize_messages(messages))
            yield handle

    @contextmanager
    def chat_span(self, model: str, context: RunContext, messages,
                  agent_name: str | None = None):
        attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": self._settings.provider,
            "gen_ai.request.model": model,
            "gen_ai.request.temperature": 0,
            "gen_ai.request.stream": True,
            "gen_ai.conversation.id": context.conversation_id,
            "enduser.id": context.user_id,
            "server.address": self._settings.server_address,
            "server.port": self._settings.server_port,
        }
        if agent_name:
            attrs["gen_ai.agent.name"] = agent_name
        with self.tracer.start_as_current_span(
            f"chat {model}", kind=SpanKind.CLIENT, attributes=attrs
        ) as span:
            handle = SpanHandle(span, self.capture_content)
            if self.capture_content:
                span.set_attribute("gen_ai.input.messages", serialize_messages(messages))
            yield handle

    def finish_chat_span(self, handle: SpanHandle, *, response_model: str,
                         response_id: str | None, finish_reason: str,
                         ttft: float | None, usage: TokenUsage, output_message) -> None:
        handle.set_attribute("gen_ai.response.model", response_model)
        if response_id:
            handle.set_attribute("gen_ai.response.id", response_id)
        handle.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
        if ttft is not None:
            handle.set_attribute("gen_ai.response.time_to_first_chunk", ttft)
        handle.set_usage(usage)
        handle.set_output([output_message])

    def record_inference(self, *, model: str, response_model: str, agent_name: str | None,
                         duration: float, usage: TokenUsage, ttft: float | None,
                         chunk_intervals: tuple[float, ...], error_type: str = "") -> None:
        attrs = self._inference_metric_attrs(model, response_model, agent_name)
        if error_type:
            failed = {**attrs, "error.type": error_type}
            self.inference_duration.record(duration, failed)
            self.inference_op_input.record(0, failed)
            self.inference_op_output.record(0, failed)
            return
        self.inference_duration.record(duration, attrs)
        if ttft is not None:
            self.inference_ttfc.record(ttft, attrs)
        for interval in chunk_intervals:
            self.inference_tpoc.record(interval, attrs)
        if usage.input_tokens:
            self.inference_input_tokens.add(
                usage.input_tokens, {**attrs, "gen_ai.token.modality": "text"})
        if usage.output_tokens:
            self.inference_output_tokens.add(
                usage.output_tokens, {**attrs, "gen_ai.token.modality": "text"})
        if usage.cache_read_tokens:
            self.inference_cache_read.add(
                usage.cache_read_tokens, {**attrs, "gen_ai.token.modality": "text"})
        if usage.cache_write_tokens:
            self.inference_cache_write.add(
                usage.cache_write_tokens, {**attrs, "gen_ai.token.modality": "text"})
        if usage.reasoning_tokens:
            self.inference_reasoning.add(
                usage.reasoning_tokens, {**attrs, "gen_ai.token.modality": "reasoning"})
        self.inference_op_input.record(usage.input_tokens, attrs)
        self.inference_op_output.record(usage.output_tokens, attrs)

    @contextmanager
    def tool_span(self, tool_name: str, span_name: str, call_id: str, args: dict,
                  *, description: str, agent_name: str, context: RunContext,
                  attrs: dict[str, Any] | None = None):
        base = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.type": "function",
            "gen_ai.agent.name": agent_name,
            "gen_ai.conversation.id": context.conversation_id,
            "enduser.id": context.user_id,
        }
        if description:
            base["gen_ai.tool.description"] = description
        if call_id:
            base["gen_ai.tool.call.id"] = call_id
        if attrs:
            base.update(attrs)
        metric_attrs = {
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.type": "function",
            "gen_ai.agent.name": agent_name,
        }
        started = time.perf_counter()
        with self.tracer.start_as_current_span(
            span_name, kind=SpanKind.INTERNAL, attributes=base
        ) as span:
            handle = SpanHandle(span, self.capture_content)
            if self.capture_content:
                span.set_attribute("gen_ai.tool.call.arguments", json_str(args))
            try:
                yield handle
            except Exception as exc:
                handle.set_error(exc)
                self.tool_duration.record(
                    time.perf_counter() - started,
                    {**metric_attrs, "error.type": type(exc).__name__},
                )
                raise
            else:
                self.tool_duration.record(time.perf_counter() - started, metric_attrs)

    @contextmanager
    def retrieval_span(self, *, data_source_id: str, top_k: int, query: str,
                       agent_name: str, context: RunContext):
        attrs = {
            "gen_ai.operation.name": "retrieval",
            "gen_ai.data_source.id": data_source_id,
            "gen_ai.retrieval.top_k": top_k,
            "gen_ai.agent.name": agent_name,
            "gen_ai.conversation.id": context.conversation_id,
            "enduser.id": context.user_id,
        }
        started = time.perf_counter()
        with self.tracer.start_as_current_span(
            f"retrieval {data_source_id}", kind=SpanKind.CLIENT, attributes=attrs
        ) as span:
            handle = SpanHandle(span, self.capture_content)
            if self.capture_content:
                span.set_attribute("gen_ai.retrieval.query.text", query)
            try:
                yield handle
            except Exception as exc:
                handle.set_error(exc)
                self.operation_duration.record(
                    time.perf_counter() - started,
                    {"gen_ai.operation.name": "retrieval",
                     "gen_ai.agent.name": agent_name,
                     "error.type": type(exc).__name__},
                )
                raise
            else:
                self.operation_duration.record(
                    time.perf_counter() - started,
                    {"gen_ai.operation.name": "retrieval",
                     "gen_ai.agent.name": agent_name},
                )

    @contextmanager
    def memory_span(self, operation: str, store_id: str, context: RunContext,
                    attrs: dict[str, Any] | None = None):
        base = {
            "gen_ai.operation.name": operation,
            "gen_ai.memory.store.id": store_id,
            "gen_ai.conversation.id": context.conversation_id,
            "enduser.id": context.user_id,
        }
        if attrs:
            base.update(attrs)
        started = time.perf_counter()
        with self.tracer.start_as_current_span(
            operation, kind=SpanKind.INTERNAL, attributes=base
        ) as span:
            handle = SpanHandle(span, self.capture_content)
            try:
                yield handle
            except Exception as exc:
                handle.set_error(exc)
                self.operation_duration.record(
                    time.perf_counter() - started,
                    {"gen_ai.operation.name": operation,
                     "error.type": type(exc).__name__},
                )
                raise
            else:
                self.operation_duration.record(
                    time.perf_counter() - started,
                    {"gen_ai.operation.name": operation},
                )

    def record_workflow_duration(self, workflow_name: str, duration: float,
                                 error_type: str = "") -> None:
        attrs = {"gen_ai.workflow.name": workflow_name}
        if error_type:
            attrs["error.type"] = error_type
        self.workflow_duration.record(duration, attrs)

    def record_agent_duration(self, agent_name: str, model: str, duration: float,
                              error_type: str = "") -> None:
        attrs = {"gen_ai.agent.name": agent_name, "gen_ai.request.model": model}
        if error_type:
            attrs["error.type"] = error_type
        self.agent_duration.record(duration, attrs)

    def record_agent_steps(self, agent_name: str, inference_calls: int,
                           tool_calls: int) -> None:
        self.agent_inference_calls.record(inference_calls, {"gen_ai.agent.name": agent_name})
        self.agent_tool_calls.record(tool_calls, {"gen_ai.agent.name": agent_name})

    def force_flush(self) -> None:
        self.tracer_provider.force_flush()
        self.meter_provider.force_flush()

    def shutdown(self) -> None:
        self.force_flush()
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()

    def finished_spans(self):
        if self.span_exporter is None:
            raise RuntimeError("telemetry is not in in-memory mode")
        return self.span_exporter.get_finished_spans()

    def clear_spans(self) -> None:
        if self.span_exporter is not None:
            self.span_exporter.clear()
