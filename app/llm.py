import json
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config import Settings
from context import current_context
from models import InferenceResult, TokenUsage
from telemetry import Telemetry


class ChatClient:
    """OpenAI-compatible streaming client that emits GenAI inference telemetry."""

    def __init__(self, settings: Settings, telemetry: Telemetry):
        self._settings = settings
        self._telemetry = telemetry
        self._clients: dict[str, ChatOpenAI] = {}

    def _client(self, model: str) -> ChatOpenAI:
        if model not in self._clients:
            self._clients[model] = ChatOpenAI(
                model=model,
                base_url=self._settings.llm_base_url,
                api_key="dummy",
                temperature=0,
                stream_usage=True,
                timeout=30,
            )
        return self._clients[model]

    def complete(self, model: str, messages, *, agent_name: str,
                 tools: list | None = None) -> InferenceResult:
        context = current_context()
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        llm = self._client(model)
        if tools:
            llm = llm.bind_tools(tools)

        started = time.perf_counter()
        with self._telemetry.chat_span(model, context, messages, agent_name=agent_name) as span:
            ttft = None
            last_chunk_at = None
            chunk_intervals: list[float] = []
            parts: list[str] = []
            usage_metadata: dict = {}
            response_model = model
            finish_reason = "stop"
            message_id = None
            tool_acc: dict[int, dict] = {}
            try:
                for chunk in llm.stream(messages):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - started
                    else:
                        chunk_intervals.append(now - last_chunk_at)
                    last_chunk_at = now
                    if getattr(chunk, "id", None):
                        message_id = chunk.id
                    if chunk.content:
                        parts.append(
                            chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                        )
                    for tool_chunk in getattr(chunk, "tool_call_chunks", None) or []:
                        index = tool_chunk.get("index")
                        acc = tool_acc.setdefault(
                            0 if index is None else index,
                            {"name": None, "args": "", "id": None},
                        )
                        if tool_chunk.get("name"):
                            acc["name"] = tool_chunk["name"]
                        if tool_chunk.get("id"):
                            acc["id"] = tool_chunk["id"]
                        if tool_chunk.get("args"):
                            acc["args"] += tool_chunk["args"]
                    if getattr(chunk, "usage_metadata", None):
                        usage_metadata = chunk.usage_metadata
                    if getattr(chunk, "response_metadata", None):
                        response_model = (
                            chunk.response_metadata.get("model_name", response_model)
                            or response_model
                        )
                        reason = chunk.response_metadata.get("finish_reason")
                        if reason:
                            finish_reason = reason
            except Exception as exc:
                self._telemetry.record_inference(
                    model=model,
                    response_model=response_model,
                    agent_name=agent_name,
                    duration=time.perf_counter() - started,
                    usage=TokenUsage(),
                    ttft=None,
                    chunk_intervals=(),
                    error_type=type(exc).__name__,
                )
                span.set_error(exc)
                raise

            tool_calls = []
            for index in sorted(tool_acc):
                acc = tool_acc[index]
                if not acc.get("name"):
                    continue
                try:
                    arguments = json.loads(acc["args"] or "{}")
                except json.JSONDecodeError:
                    arguments = {"_raw": acc["args"]}
                tool_calls.append({
                    "name": acc["name"],
                    "args": arguments,
                    "id": acc["id"] or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "tool_call",
                })

            message = AIMessage(
                content="".join(parts),
                tool_calls=tool_calls,
                id=message_id,
            )
            usage = TokenUsage.from_metadata(usage_metadata)
            duration = time.perf_counter() - started
            time_per_chunk = (
                sum(chunk_intervals) / len(chunk_intervals) if chunk_intervals else None
            )
            self._telemetry.finish_chat_span(
                span,
                response_model=response_model,
                response_id=message_id,
                finish_reason=finish_reason,
                ttft=ttft,
                usage=usage,
                output_message=message,
            )
            self._telemetry.record_inference(
                model=model,
                response_model=response_model,
                agent_name=agent_name,
                duration=duration,
                usage=usage,
                ttft=ttft,
                chunk_intervals=tuple(chunk_intervals),
            )
            return InferenceResult(
                message=message,
                usage=usage,
                ttft=ttft,
                time_per_chunk=time_per_chunk,
            )
