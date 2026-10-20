# GenAI Semantic-Conventions Conformance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the LangGraph demo a clean, real reference implementation of the official GenAI telemetry standard: add Agent Skills, retrieval, and memory; remove every `poc.*` signal; align both provisioned dashboards and the Tempo/Prometheus configuration.

**Architecture:** Keep the supervisor graph and add a `policy_agent` that consumes a real local Agent Skill (load / read resource / run script) plus local retrieval. Put memory read/write at workflow boundaries. Split instrumentation into small modules (`context`, `telemetry`, `skills`, `memory`) and leave the graph/prompts in `langgraph_app.py`. Cost is derived inline in Grafana from token counters; no cost metric and no recording rule.

**Tech Stack:** Python 3.10, LangGraph, LangChain OpenAI client, OpenTelemetry SDK, OTLP -> Collector -> Tempo + Prometheus -> Grafana 13. Tests use `unittest` and OTel in-memory exporters.

---

## Standard precedence (read first)

The official documents are the only source of truth and must not be edited:
`ai-observability-design.md` and `ai-observability-standard.zh.md`.

Where the earlier implementation spec (`docs/superpowers/specs/2026-10-19-genai-semconv-conformance-design.md`) disagrees with the official standard, the official standard wins:

1. `invoke_workflow` carries **no** `gen_ai.usage.*` totals (standard 8.2 / Appendix A row 1).
2. `gen_ai.conversation.id` and `enduser.id` are **never** metric dimensions, including Tempo span-metrics (R-11.1-01 / R-8.3-02). Conversation/user views come from TraceQL tables.
3. No new Prometheus metric or recording rule (R-8.4-04). Cost is derived inline in the Grafana token-cost queries from token counters × the deployment price table.
4. `gen_ai.tool.definitions` is in the Opt-In content group; it is gated by the content switch along with messages and tool arguments/results.
5. `chat` does not carry `gen_ai.agent.name`; the parent `invoke_agent` identifies the agent. Inference metrics still carry `gen_ai.agent.name` because 8.3 requires it.

## Repository state

- The repository root is **not** a git repository. Do not initialize git and do not commit.
- Checkpoints are test runs and JSON/config validation.
- The Docker stack is running; the compose bridge may need its gateway address after a host reboot:
  `ip addr add 172.18.0.1/16 dev br-7baf58fc3c9c`
- The local LLM gateway on `127.0.0.1:4141` is currently down. Tasks 1-7 do not need it; Task 8's full run does.

## File map

| File | Responsibility |
|---|---|
| `app/context.py` (new) | ContextVars for enduser id, conversation id, current agent |
| `app/telemetry.py` (new) | OTel providers, instruments, views, content switch, serialization, execute-tool span helper |
| `app/skills.py` (new) | Skill loading, resource confinement, TF-IDF retrieval, script execution, instrumented skill tools |
| `app/memory.py` (new) | JSON memory store, atomic writes, `search_memory` / `create_memory` spans and metrics |
| `app/langgraph_app.py` (rewrite) | Graph, prompts, LLM instrumentation, conversation scripts, run loop |
| `app/skills/travel_policy/SKILL.md` (new) | Agent Skill metadata + instructions |
| `app/skills/travel_policy/references/*.md` (new) | Entry requirements, baggage, budget guide |
| `app/skills/travel_policy/scripts/estimate_budget.py` (new) | Real budget estimator |
| `app/tests/test_telemetry.py` (new) | Content switch + serialization |
| `app/tests/test_skills.py` (new) | Skill spans, retrieval span, command span, path confinement |
| `app/tests/test_memory.py` (new) | Memory spans, store behavior, malformed file recovery |
| `app/tests/test_workflow.py` (new) | Workflow root has no usage/poc attrs; memory spans exist |
| `grafana/dashboards/agent-insight.json` | Standard KPI/terms, trace-based conversations, Scenarios row |
| `grafana/dashboards/token-cost.json` | Cost derived inline from tokens × prices |
| `tempo/tempo.yaml` | Add bounded skill/retrieval/memory/error dimensions; remove conversation/user |
| `README.md` | New signal, scenario, conformance, and run documentation |

No changes to `prometheus/prometheus.yml`, `docker-compose.yml`, or a new `prometheus/rules/` file.

---

### Task 1: Context and telemetry foundation

**Files:**
- Create: `app/context.py`
- Create: `app/telemetry.py`
- Create: `app/tests/__init__.py`
- Test: `app/tests/test_telemetry.py`

- [ ] **Step 1: Write the failing tests**

```python
# app/tests/test_telemetry.py
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["POC_TELEMETRY_IN_MEMORY"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import telemetry


class TelemetryTest(unittest.TestCase):
    def test_content_switch_reads_env_dynamically(self):
        with patch.dict(os.environ, {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
            self.assertTrue(telemetry.capture_content())
        with patch.dict(os.environ, {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": ""}):
            self.assertFalse(telemetry.capture_content())

    def test_serialize_messages_uses_spec_shape(self):
        from langchain_core.messages import HumanMessage

        raw = telemetry.serialize_messages([HumanMessage(content="weather?")])
        self.assertEqual(
            json.loads(raw),
            [{"role": "user", "parts": [{"type": "text", "content": "weather?"}]}],
        )

    def test_instruments_exist(self):
        for name in (
            "inference_duration",
            "inference_ttfc",
            "inference_tpoc",
            "operation_duration",
            "tool_duration",
            "agent_duration",
        ):
            self.assertTrue(hasattr(telemetry, name), name)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest app.tests.test_telemetry -v`
Expected: ERROR/FAIL because `app.telemetry` does not exist.

- [ ] **Step 3: Write `app/context.py`**

```python
import contextvars

_enduser_id = contextvars.ContextVar("enduser_id", default="")
_conversation_id = contextvars.ContextVar("conversation_id", default="")
_current_agent = contextvars.ContextVar("current_agent", default="")


def set_run_context(enduser_id: str, conversation_id: str) -> None:
    _enduser_id.set(enduser_id)
    _conversation_id.set(conversation_id)


def get_enduser_id() -> str:
    return _enduser_id.get()


def get_conversation_id() -> str:
    return _conversation_id.get()


def get_current_agent() -> str:
    return _current_agent.get()


def set_current_agent(name: str):
    return _current_agent.set(name)


def reset_current_agent(token) -> None:
    _current_agent.reset(token)
```

- [ ] **Step 4: Write `app/telemetry.py`**

```python
import json
import os
import time
from contextlib import contextmanager

from opentelemetry import metrics, trace
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
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode

from context import get_conversation_id, get_current_agent, get_enduser_id

OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
IN_MEMORY = os.getenv("POC_TELEMETRY_IN_MEMORY", "").lower() in ("1", "true", "yes")

resource = Resource.create({
    "service.name": "langgraph-agent",
    "service.version": "4.0.0",
    "deployment.environment": "local",
})

provider = TracerProvider(resource=resource)
span_exporter = None
if IN_MEMORY:
    span_exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
else:
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True))
    )
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("gen_ai.langgraph")

if IN_MEMORY:
    metric_reader = InMemoryMetricReader()
else:
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            endpoint=OTLP_ENDPOINT,
            insecure=True,
            preferred_temporality={
                Counter: AggregationTemporality.CUMULATIVE,
                Histogram: AggregationTemporality.CUMULATIVE,
                UpDownCounter: AggregationTemporality.CUMULATIVE,
            },
        ),
        export_interval_millis=5000,
    )

TOKEN_BUCKETS = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
VIEWS = [
    View(instrument_type=Histogram, instrument_name="gen_ai.execute_tool.duration",
         aggregation=ExplicitBucketHistogramAggregation(
             [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])),
    View(instrument_type=Histogram, instrument_name="gen_ai.client.inference.time_to_first_chunk",
         aggregation=ExplicitBucketHistogramAggregation(
             [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30])),
    View(instrument_type=Histogram, instrument_name="gen_ai.client.inference.time_per_output_chunk",
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
    View(instrument_type=Histogram, instrument_name="gen_ai.client.inference.operation.input_tokens",
         aggregation=ExplicitBucketHistogramAggregation(TOKEN_BUCKETS)),
    View(instrument_type=Histogram, instrument_name="gen_ai.client.inference.operation.output_tokens",
         aggregation=ExplicitBucketHistogramAggregation(TOKEN_BUCKETS)),
]

meter_provider = MeterProvider(
    resource=resource, metric_readers=[metric_reader], views=VIEWS
)
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter("gen_ai.langgraph")

# Descriptions must match across every producer of a metric name.
inference_duration = meter.create_histogram(
    "gen_ai.client.inference.duration", unit="s", description="GenAI inference operation duration")
inference_ttfc = meter.create_histogram(
    "gen_ai.client.inference.time_to_first_chunk", unit="s", description="Time to first streamed chunk")
inference_tpoc = meter.create_histogram(
    "gen_ai.client.inference.time_per_output_chunk", unit="s", description="Time per output chunk")
inference_input_tokens = meter.create_counter(
    "gen_ai.client.inference.usage.input_tokens", unit="{token}", description="Number of input tokens used")
inference_output_tokens = meter.create_counter(
    "gen_ai.client.inference.usage.output_tokens", unit="{token}", description="Number of output tokens used")
inference_cache_read = meter.create_counter(
    "gen_ai.client.inference.usage.cache_read.input_tokens", unit="{token}", description="Number of cached input tokens read")
inference_cache_write = meter.create_counter(
    "gen_ai.client.inference.usage.cache_write.input_tokens", unit="{token}", description="Number of input tokens written to cache")
inference_reasoning = meter.create_counter(
    "gen_ai.client.inference.usage.reasoning.output_tokens", unit="{token}", description="Number of reasoning output tokens used")
inference_op_input = meter.create_histogram(
    "gen_ai.client.inference.operation.input_tokens", unit="{token}", description="Input tokens per inference operation")
inference_op_output = meter.create_histogram(
    "gen_ai.client.inference.operation.output_tokens", unit="{token}", description="Output tokens per inference operation")
operation_duration = meter.create_histogram(
    "gen_ai.client.operation.duration", unit="s", description="GenAI operation duration")
workflow_duration = meter.create_histogram(
    "gen_ai.invoke_workflow.duration", unit="s", description="GenAI workflow duration")
agent_duration = meter.create_histogram(
    "gen_ai.invoke_agent.duration", unit="s", description="GenAI agent invocation duration")
agent_inference_calls = meter.create_histogram(
    "gen_ai.invoke_agent.inference_calls", unit="{inference_call}", description="Inference calls per agent invocation")
agent_tool_calls = meter.create_histogram(
    "gen_ai.invoke_agent.tool_calls", unit="{tool_call}", description="Tool calls per agent invocation")
tool_duration = meter.create_histogram(
    "gen_ai.execute_tool.duration", unit="s", description="GenAI tool execution duration")


def capture_content() -> bool:
    value = os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "")
    return value.strip().lower() in ("1", "true", "yes", "on")


_ROLE = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def json_str(value) -> str:
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


def record_operation_duration(operation: str, duration: float, *, agent_name: str = "",
                              error_type: str = "") -> None:
    attrs = {"gen_ai.operation.name": operation}
    if agent_name:
        attrs["gen_ai.agent.name"] = agent_name
    if error_type:
        attrs["error.type"] = error_type
    operation_duration.record(duration, attrs)


@contextmanager
def execute_tool_span(tool_name: str, span_name: str, call_id: str, args: dict, *,
                      description: str = "", attrs: dict | None = None):
    agent_name = get_current_agent()
    conversation_id = get_conversation_id()
    enduser_id = get_enduser_id()
    base = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": tool_name,
        "gen_ai.tool.type": "function",
    }
    if description:
        base["gen_ai.tool.description"] = description
    if call_id:
        base["gen_ai.tool.call.id"] = call_id
    if agent_name:
        base["gen_ai.agent.name"] = agent_name
    if conversation_id:
        base["gen_ai.conversation.id"] = conversation_id
    if enduser_id:
        base["enduser.id"] = enduser_id
    if attrs:
        base.update(attrs)
    metric_attrs = {"gen_ai.tool.name": tool_name, "gen_ai.tool.type": "function"}
    if agent_name:
        metric_attrs["gen_ai.agent.name"] = agent_name
    started = time.perf_counter()
    with tracer.start_as_current_span(
        span_name, kind=SpanKind.INTERNAL, attributes=base
    ) as span:
        if capture_content():
            span.set_attribute("gen_ai.tool.call.arguments", json_str(args))
        try:
            yield span
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            tool_duration.record(
                time.perf_counter() - started,
                {**metric_attrs, "error.type": type(exc).__name__},
            )
            raise
        else:
            tool_duration.record(time.perf_counter() - started, metric_attrs)


def get_finished_spans():
    if span_exporter is None:
        raise RuntimeError("in-memory mode is not enabled")
    return span_exporter.get_finished_spans()


def clear_spans() -> None:
    if span_exporter is not None:
        span_exporter.clear()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m unittest app.tests.test_telemetry -v`
Expected: 3 tests PASS.

- [ ] **Step 6: Checkpoint**

Run: `python3 -m py_compile app/context.py app/telemetry.py`
Expected: no output, exit 0.

---

### Task 2: Agent Skill package and skill tools

**Files:**
- Create: `app/skills/travel_policy/SKILL.md`
- Create: `app/skills/travel_policy/references/entry_requirements.md`
- Create: `app/skills/travel_policy/references/baggage_rules.md`
- Create: `app/skills/travel_policy/references/budget_guide.md`
- Create: `app/skills/travel_policy/scripts/estimate_budget.py`
- Create: `app/skills.py`
- Test: `app/tests/test_skills.py`

- [ ] **Step 1: Create the skill files**

`app/skills/travel_policy/SKILL.md`:

```markdown
---
name: travel_policy
description: Entry requirements, baggage rules, and daily budget guidance for common travel destinations.
---

# Travel Policy

Use this skill when the traveler asks about visas, entry requirements, baggage
allowances, or a reasonable daily budget.

## Procedure

1. Search the references for the destination and topic:
   `search_skill_resources(skill_name="travel_policy", query="Japan visa entry requirements")`.
2. Read the most relevant reference in full:
   `read_skill_resource(skill_name="travel_policy", resource_name="references/entry_requirements.md")`.
3. When the traveler asks for a budget, run:
   `run_skill_script(skill_name="travel_policy", script_name="scripts/estimate_budget.py", args=["--city","Tokyo","--days","5","--style","mid-range"])`.
4. Answer only from the skill content and tool results. If a fact is missing,
   say that it could not be verified.
```

`references/entry_requirements.md`:

```markdown
# Entry requirements

- Japan: visa-free short stays for many passports; confirm the passport's
  nationality with the nearest embassy. Passport validity of at least six
  months is recommended. A return ticket and proof of accommodation may be
  requested at the border.
- Portugal: Schengen short-stay rules apply for many passports (90 days in
  any 180-day period). Check the passport's nationality before travel.
- Canada: an eTA is commonly required for visa-exempt air travelers; other
  passports need a visitor visa. Confirm the current rule with the embassy.

These notes are guidance, not legal advice. Always verify with the destination's
official immigration authority before booking.
```

`references/baggage_rules.md`:

```markdown
# Baggage guidance

- Economy short-haul: one cabin bag plus one personal item is common; checked
  allowance is usually 20-23 kg and varies by fare.
- Economy long-haul: one checked bag of 23 kg is common; some fares exclude
  checked baggage.
- Always check the operating carrier and fare class; codeshare rules can differ.

This guide is generic; the airline's fare rules are authoritative.
```

`references/budget_guide.md`:

```markdown
# Daily budget guide

The estimate_budget.py script returns a daily range by city and style:

- `budget`: hostel or simple guesthouse, local transport, casual meals.
- `mid-range`: 3-star hotel, some taxis, one sit-down meal per day.
- `comfort`: 4-star hotel, more taxis, restaurant meals.

The ranges are planning estimates in USD and exclude international flights.
```

`scripts/estimate_budget.py`:

```python
#!/usr/bin/env python3
"""Estimate a daily and total travel budget from city, days, and style."""
import argparse
import json

DAILY = {
    "tokyo": {"budget": (70, 110), "mid-range": (140, 220), "comfort": (260, 420)},
    "lisbon": {"budget": (55, 90), "mid-range": (110, 180), "comfort": (210, 340)},
    "toronto": {"budget": (75, 120), "mid-range": (150, 240), "comfort": (280, 450)},
    "berlin": {"budget": (60, 95), "mid-range": (120, 190), "comfort": (220, 360)},
}
DEFAULT_DAILY = {"budget": (60, 100), "mid-range": (120, 200), "comfort": (220, 380)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--style", choices=["budget", "mid-range", "comfort"], default="mid-range")
    args = parser.parse_args()
    city = args.city.strip().lower()
    low, high = DAILY.get(city, DEFAULT_DAILY)[args.style]
    result = {
        "city": args.city.strip(),
        "days": args.days,
        "style": args.style,
        "currency": "USD",
        "daily_low": low,
        "daily_high": high,
        "total_low": low * args.days,
        "total_high": high * args.days,
        "note": "Planning estimate; excludes international flights.",
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing tests**

```python
# app/tests/test_skills.py
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["POC_TELEMETRY_IN_MEMORY"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import telemetry
import skills
from context import set_run_context


class SkillTest(unittest.TestCase):
    def setUp(self):
        telemetry.clear_spans()
        set_run_context("user-test", "conv-test")

    def _span(self, name):
        return next(span for span in telemetry.get_finished_spans() if span.name == name)

    def test_load_skill_emits_refined_span(self):
        text = skills.load_skill("travel_policy")
        self.assertIn("travel_policy", text)
        span = self._span("execute_tool load_skill travel_policy")
        self.assertEqual(span.attributes["gen_ai.operation.name"], "execute_tool")
        self.assertEqual(span.attributes["gen_ai.skill.name"], "travel_policy")
        self.assertIn("gen_ai.skill.source.uri", span.attributes)
        self.assertNotIn("gen_ai.tool.call.arguments", span.attributes)

    def test_content_switch_enables_tool_arguments(self):
        with patch.dict(os.environ, {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
            skills.load_skill("travel_policy")
        span = self._span("execute_tool load_skill travel_policy")
        self.assertIn("gen_ai.tool.call.arguments", span.attributes)

    def test_read_resource_rejects_traversal(self):
        with self.assertRaises(ValueError):
            skills.read_skill_resource("travel_policy", "../SKILL.md")

    def test_search_emits_tool_and_retrieval_spans(self):
        payload = json.loads(skills.search_skill_resources("travel_policy", "Japan visa", top_k=2))
        ids = [doc["id"] for doc in payload["documents"]]
        self.assertIn("references/entry_requirements.md", ids)
        tool = self._span("execute_tool search_skill_resources")
        retrieval = self._span("retrieval travel-policy-docs")
        self.assertEqual(tool.attributes["gen_ai.tool.name"], "search_skill_resources")
        self.assertEqual(retrieval.attributes["gen_ai.operation.name"], "retrieval")
        self.assertEqual(retrieval.attributes["gen_ai.data_source.id"], "travel-policy-docs")
        self.assertEqual(retrieval.attributes["gen_ai.retrieval.top_k"], 2)
        self.assertNotIn("gen_ai.retrieval.documents", retrieval.attributes)

    def test_script_execution_emits_command_span(self):
        out = skills.run_skill_script(
            "travel_policy",
            "scripts/estimate_budget.py",
            ["--city", "Tokyo", "--days", "5", "--style", "mid-range"],
        )
        payload = json.loads(out)
        self.assertEqual(payload["total_low"], payload["daily_low"] * 5)
        span = self._span(
            "execute_tool run_skill_script travel_policy scripts/estimate_budget.py"
        )
        self.assertEqual(span.attributes["gen_ai.skill.resource.name"], "scripts/estimate_budget.py")
        self.assertIn("process.executable.name", span.attributes)
        self.assertEqual(span.attributes["process.exit.code"], 0)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python3 -m unittest app.tests.test_skills -v`
Expected: FAIL because `app.skills` does not exist.

- [ ] **Step 4: Write `app/skills.py`**

```python
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Annotated

import yaml
from langchain_core.tools import InjectedToolCallId
from opentelemetry.trace import SpanKind, Status, StatusCode

from context import get_conversation_id, get_current_agent, get_enduser_id
from telemetry import (
    capture_content,
    execute_tool_span,
    json_str,
    record_operation_duration,
    tracer,
)

SKILLS_ROOT = Path(__file__).resolve().parent / "skills"
DATA_SOURCE_ID = "travel-policy-docs"
SCRIPT_TIMEOUT = 20


def _skill_dir(skill_name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9_]+", skill_name or ""):
        raise ValueError(f"invalid skill name: {skill_name!r}")
    root = SKILLS_ROOT.resolve()
    path = (root / skill_name).resolve()
    if root not in path.parents:
        raise ValueError("skill path escapes the skills root")
    if not path.is_dir():
        raise FileNotFoundError(f"skill not found: {skill_name}")
    return path


def _parse_skill_md(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError("SKILL.md must start with a frontmatter block")
    _, frontmatter, body = text.split("---", 2)
    meta = yaml.safe_load(frontmatter) or {}
    if not isinstance(meta, dict) or not meta.get("name"):
        raise ValueError("SKILL.md frontmatter must contain a name")
    return meta, body.strip()


def _skill_meta(skill_name: str) -> tuple[Path, dict, str]:
    directory = _skill_dir(skill_name)
    meta, body = _parse_skill_md(directory / "SKILL.md")
    return directory, meta, body


def _safe_resource(skill_name: str, resource_name: str) -> Path:
    directory = _skill_dir(skill_name)
    candidate = Path(resource_name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("resource must be a relative path inside the skill")
    path = (directory / candidate).resolve()
    if directory.resolve() not in path.parents:
        raise ValueError("resource path escapes the skill directory")
    if not path.is_file():
        raise FileNotFoundError(f"resource not found: {resource_name}")
    return path


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _search(skill_name: str, query: str, top_k: int) -> list[dict]:
    directory, _, _ = _skill_meta(skill_name)
    docs = []
    for path in sorted((directory / "references").rglob("*.md")):
        rel = path.relative_to(directory).as_posix()
        docs.append((rel, _tokenize(path.read_text(encoding="utf-8"))))
    query_tokens = _tokenize(query)
    if not docs or not query_tokens:
        return []
    df = Counter()
    for _, tokens in docs:
        df.update(set(tokens))
    idf = {term: math.log((1 + len(docs)) / (1 + count)) + 1 for term, count in df.items()}

    def score(tokens: list[str]) -> float:
        counts = Counter(tokens)
        query_counts = Counter(query_tokens)
        numerator = sum(query_counts[t] * counts.get(t, 0) * idf.get(t, 1) ** 2 for t in query_counts)
        left = math.sqrt(sum((query_counts[t] * idf.get(t, 1)) ** 2 for t in query_counts))
        right = math.sqrt(sum((counts[t] * idf.get(t, 1)) ** 2 for t in counts))
        return numerator / (left * right) if left and right else 0.0

    ranked = sorted(
        ((score(tokens), rel) for rel, tokens in docs),
        key=lambda item: item[0],
        reverse=True,
    )
    out = []
    for value, rel in ranked[:max(1, top_k)]:
        text = (directory / rel).read_text(encoding="utf-8")
        out.append({"id": rel, "score": round(value, 4), "snippet": text[:500]})
    return out


def load_skill(skill_name: str, tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
    """Load an Agent Skill into the agent context and return its instructions."""
    directory, meta, body = _skill_meta(skill_name)
    with execute_tool_span(
        "load_skill",
        f"execute_tool load_skill {skill_name}",
        tool_call_id,
        {"skill_name": skill_name},
        description=load_skill.__doc__,
        attrs={
            "gen_ai.skill.name": skill_name,
            "gen_ai.skill.description": meta.get("description", ""),
            "gen_ai.skill.source.uri": directory.as_uri(),
        },
    ) as span:
        result = f"# {meta['name']}\n\n{body}"
        if capture_content():
            span.set_attribute("gen_ai.tool.call.result", result)
        return result


def read_skill_resource(skill_name: str, resource_name: str,
                        tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
    """Read a reference, asset, or script file bundled with an Agent Skill."""
    path = _safe_resource(skill_name, resource_name)
    directory, meta, _ = _skill_meta(skill_name)
    with execute_tool_span(
        "read_skill_resource",
        f"execute_tool read_skill_resource {skill_name} {resource_name}",
        tool_call_id,
        {"skill_name": skill_name, "resource_name": resource_name},
        description=read_skill_resource.__doc__,
        attrs={
            "gen_ai.skill.name": skill_name,
            "gen_ai.skill.resource.name": resource_name,
            "gen_ai.skill.description": meta.get("description", ""),
            "gen_ai.skill.source.uri": directory.as_uri(),
        },
    ) as span:
        result = path.read_text(encoding="utf-8")
        if capture_content():
            span.set_attribute("gen_ai.tool.call.result", result)
        return result


def search_skill_resources(skill_name: str, query: str, top_k: int = 5,
                           tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
    """Search the skill's reference documents and return the top matching snippets."""
    with execute_tool_span(
        "search_skill_resources",
        "execute_tool search_skill_resources",
        tool_call_id,
        {"skill_name": skill_name, "query": query, "top_k": top_k},
        description=search_skill_resources.__doc__,
        attrs={"gen_ai.skill.name": skill_name},
    ) as tool_span:
        started = time.perf_counter()
        retrieval_attrs = {
            "gen_ai.operation.name": "retrieval",
            "gen_ai.data_source.id": DATA_SOURCE_ID,
            "gen_ai.retrieval.top_k": top_k,
        }
        if get_conversation_id():
            retrieval_attrs["gen_ai.conversation.id"] = get_conversation_id()
        if get_enduser_id():
            retrieval_attrs["enduser.id"] = get_enduser_id()
        with tracer.start_as_current_span(
            f"retrieval {DATA_SOURCE_ID}", kind=SpanKind.CLIENT, attributes=retrieval_attrs
        ) as span:
            if capture_content():
                span.set_attribute("gen_ai.retrieval.query.text", query)
            try:
                docs = _search(skill_name, query, top_k)
            except Exception as exc:
                span.set_attribute("error.type", type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                record_operation_duration(
                    "retrieval", time.perf_counter() - started,
                    agent_name=get_current_agent(), error_type=type(exc).__name__,
                )
                raise
            if capture_content():
                span.set_attribute(
                    "gen_ai.retrieval.documents",
                    json_str([{"id": doc["id"], "score": doc["score"]} for doc in docs]),
                )
            record_operation_duration(
                "retrieval", time.perf_counter() - started,
                agent_name=get_current_agent(),
            )
        payload = json_str({"documents": docs})
        if capture_content():
            tool_span.set_attribute("gen_ai.tool.call.result", payload)
        return payload


def run_skill_script(skill_name: str, script_name: str, args: list[str],
                     tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
    """Execute a script bundled with an Agent Skill and return its stdout."""
    path = _safe_resource(skill_name, script_name)
    if path.suffix != ".py" or "scripts" not in path.relative_to(_skill_dir(skill_name)).parts:
        raise ValueError("only scripts/*.py may be executed")
    directory, meta, _ = _skill_meta(skill_name)
    with execute_tool_span(
        "run_skill_script",
        f"execute_tool run_skill_script {skill_name} {script_name}",
        tool_call_id,
        {"skill_name": skill_name, "script_name": script_name, "args": args},
        description=run_skill_script.__doc__,
        attrs={
            "gen_ai.skill.name": skill_name,
            "gen_ai.skill.resource.name": script_name,
            "gen_ai.skill.description": meta.get("description", ""),
            "gen_ai.skill.source.uri": directory.as_uri(),
            "process.executable.name": Path(sys.executable).name,
            "process.executable.path": sys.executable,
        },
    ) as span:
        completed = subprocess.run(
            [sys.executable, str(path), *args],
            cwd=directory, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT, shell=False,
        )
        span.set_attribute("process.exit.code", completed.returncode)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"script exited {completed.returncode}")
        result = completed.stdout.strip()
        if capture_content():
            span.set_attribute("gen_ai.tool.call.result", result)
        return result
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m unittest app.tests.test_skills -v`
Expected: 5 tests PASS.

- [ ] **Step 6: Checkpoint**

Run: `python3 -m py_compile app/skills.py app/skills/travel_policy/scripts/estimate_budget.py`
Expected: no output, exit 0.

---

### Task 3: Memory store and memory spans

**Files:**
- Create: `app/memory.py`
- Test: `app/tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
# app/tests/test_memory.py
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["POC_TELEMETRY_IN_MEMORY"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory
import telemetry
from context import set_run_context


class MemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "memory.json"
        os.environ["POC_MEMORY_PATH"] = str(self.path)
        telemetry.clear_spans()
        set_run_context("user-1", "conv-1")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("POC_MEMORY_PATH", None)

    def _span(self, name):
        return next(span for span in telemetry.get_finished_spans() if span.name == name)

    def test_create_then_search(self):
        record_id = memory.create_memory("question", "answer")
        self.assertTrue(record_id.startswith("mem_"))
        records = memory.search_memory("question")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["content"]["answer"], "answer")
        create = self._span("create_memory")
        search = self._span("search_memory")
        self.assertEqual(create.attributes["gen_ai.memory.record.count"], 1)
        self.assertEqual(create.attributes["gen_ai.memory.record.id"], record_id)
        self.assertEqual(search.attributes["gen_ai.memory.record.count"], 1)
        self.assertEqual(search.attributes["gen_ai.memory.store.id"], "traveler_memory")
        self.assertNotIn("gen_ai.memory.query.text", search.attributes)
        self.assertNotIn("gen_ai.memory.records", search.attributes)

    def test_search_recovers_from_malformed_file(self):
        self.path.write_text("{not json", encoding="utf-8")
        records = memory.search_memory("question")
        self.assertEqual(records, [])
        span = self._span("search_memory")
        self.assertIn("error.type", span.attributes)

    def test_records_are_scoped_to_conversation(self):
        memory.create_memory("q1", "a1")
        memory.search_memory("q1")
        set_run_context("user-2", "conv-2")
        records = memory.search_memory("q1")
        self.assertEqual(records, [])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest app.tests.test_memory -v`
Expected: FAIL because `app.memory` does not exist.

- [ ] **Step 3: Write `app/memory.py`**

```python
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from opentelemetry.trace import SpanKind, Status, StatusCode

from context import get_conversation_id, get_enduser_id
from telemetry import (
    capture_content,
    json_str,
    record_operation_duration,
    tracer,
)

STORE_ID = "traveler_memory"
DEFAULT_PATH = Path(__file__).resolve().parent / ".state" / "memory.json"


def _store_path() -> Path:
    return Path(os.getenv("POC_MEMORY_PATH", str(DEFAULT_PATH)))


def _read_store() -> dict:
    path = _store_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_store(store: dict) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _span_attrs(operation: str) -> dict:
    attrs = {
        "gen_ai.operation.name": operation,
        "gen_ai.memory.store.id": STORE_ID,
    }
    conversation_id = get_conversation_id()
    enduser_id = get_enduser_id()
    if conversation_id:
        attrs["gen_ai.conversation.id"] = conversation_id
    if enduser_id:
        attrs["enduser.id"] = enduser_id
    return attrs


def search_memory(query: str, limit: int = 5) -> list[dict]:
    started = time.perf_counter()
    with tracer.start_as_current_span(
        "search_memory", kind=SpanKind.INTERNAL, attributes=_span_attrs("search_memory")
    ) as span:
        if capture_content():
            span.set_attribute("gen_ai.memory.query.text", query)
        try:
            store = _read_store()
            conversation_id = get_conversation_id()
            records = [
                record for record in store.get(STORE_ID, [])
                if record.get("metadata", {}).get("conversation_id") == conversation_id
            ]
            records = sorted(records, key=lambda r: r.get("metadata", {}).get("created_at", ""))[-limit:]
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            span.set_attribute("gen_ai.memory.record.count", 0)
            record_operation_duration(
                "search_memory", time.perf_counter() - started, error_type=type(exc).__name__)
            return []
        span.set_attribute("gen_ai.memory.record.count", len(records))
        if capture_content():
            span.set_attribute("gen_ai.memory.records", json_str(records))
        record_operation_duration("search_memory", time.perf_counter() - started)
        return records


def create_memory(question: str, answer: str) -> str:
    started = time.perf_counter()
    record_id = f"mem_{uuid.uuid4().hex[:16]}"
    record = {
        "id": record_id,
        "content": {"question": question, "answer": answer},
        "metadata": {
            "conversation_id": get_conversation_id(),
            "user_id": get_enduser_id(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    with tracer.start_as_current_span(
        "create_memory", kind=SpanKind.INTERNAL, attributes=_span_attrs("create_memory")
    ) as span:
        try:
            store = _read_store()
            store.setdefault(STORE_ID, []).append(record)
            _write_store(store)
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            span.set_attribute("gen_ai.memory.record.count", 1)
            span.set_attribute("gen_ai.memory.record.id", record_id)
            record_operation_duration(
                "create_memory", time.perf_counter() - started, error_type=type(exc).__name__)
            return ""
        span.set_attribute("gen_ai.memory.record.count", 1)
        span.set_attribute("gen_ai.memory.record.id", record_id)
        if capture_content():
            span.set_attribute("gen_ai.memory.records", json_str([record]))
        record_operation_duration("create_memory", time.perf_counter() - started)
        return record_id


def memory_context_message(records: list[dict]):
    if not records:
        return None
    from langchain_core.messages import SystemMessage

    lines = ["Conversation memory from earlier runs:"]
    for record in records:
        content = record.get("content", {})
        lines.append(f"- Q: {content.get('question', '')}\n  A: {content.get('answer', '')}")
    return SystemMessage(content="\n".join(lines))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest app.tests.test_memory -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Checkpoint**

Run: `python3 -m py_compile app/memory.py`
Expected: no output, exit 0.

---

### Task 4: Graph integration, policy agent, and run loop

**Files:**
- Modify: `app/langgraph_app.py` (rewrite; keep the existing prompt text and tool implementations where noted)
- Test: `app/tests/test_workflow.py`

- [ ] **Step 1: Write the failing workflow conformance test**

```python
# app/tests/test_workflow.py
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["POC_TELEMETRY_IN_MEMORY"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import langgraph_app
import telemetry


class FakeApp:
    def invoke(self, state):
        return {"answer": "ok", "tool_calls": 0, "inference_calls": 0}


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["POC_MEMORY_PATH"] = str(Path(self.tmp.name) / "memory.json")
        telemetry.clear_spans()

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("POC_MEMORY_PATH", None)

    def test_workflow_root_conformance(self):
        langgraph_app.run_once(FakeApp(), 1)
        spans = telemetry.get_finished_spans()
        root = next(span for span in spans if span.name == "invoke_workflow travel_workflow")
        self.assertEqual(root.attributes["gen_ai.operation.name"], "invoke_workflow")
        self.assertEqual(root.attributes["gen_ai.workflow.name"], "travel_workflow")
        self.assertIn("gen_ai.conversation.id", root.attributes)
        self.assertIn("enduser.id", root.attributes)
        self.assertNotIn("gen_ai.usage.input_tokens", root.attributes)
        self.assertNotIn("gen_ai.usage.output_tokens", root.attributes)
        self.assertFalse(any(key.startswith("poc.") for key in root.attributes))
        self.assertTrue(any(span.name == "search_memory" for span in spans))
        self.assertTrue(any(span.name == "create_memory" for span in spans))

    def test_route_parser_knows_policy_agent(self):
        self.assertEqual(langgraph_app._parse_route("policy_agent"), "policy_agent")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest app.tests.test_workflow -v`
Expected: FAIL because the current `run_once` sets `gen_ai.usage.*` on the root and has no memory spans.

- [ ] **Step 3: Rewrite `app/langgraph_app.py`**

Replace the module with the code below. The prompt texts (`HOUSE_STYLE_BLOCK`, `SYSTEM_PROMPT`,
`SUPERVISOR_HINT`, `WEATHER_HINT`, `CURRENCY_HINT`, `WRITER_HINT`) are retained with the
policy additions noted. The weather/currency functions keep their real HTTP behavior.

```python
"""LangGraph travel assistant instrumented per the OpenTelemetry GenAI conventions.

Scenarios: supervisor routing, agentic tools, Agent Skills (load / read resource /
run script), local retrieval, and cross-run memory. All work is real: real model
calls, real HTTP tools, real files, real subprocess scripts.
"""

import concurrent.futures
import json
import os
import time
import urllib.parse
import urllib.request
import uuid

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import InjectedToolCallId
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from opentelemetry.trace import SpanKind, Status, StatusCode
from typing import Annotated, TypedDict
from urllib.parse import urlparse

import telemetry
from context import (
    get_conversation_id,
    get_enduser_id,
    reset_current_agent,
    set_current_agent,
    set_run_context,
)
from memory import create_memory, memory_context_message, search_memory
from skills import (
    load_skill,
    read_skill_resource,
    run_skill_script,
    search_skill_resources,
)
from telemetry import (
    agent_duration,
    agent_inference_calls,
    agent_tool_calls,
    capture_content,
    execute_tool_span,
    inference_cache_read,
    inference_cache_write,
    inference_duration,
    inference_input_tokens,
    inference_op_input,
    inference_op_output,
    inference_output_tokens,
    inference_reasoning,
    inference_tpoc,
    inference_ttfc,
    messages_json,
    record_operation_duration,
    serialize_messages,
    tracer,
    workflow_duration,
)

OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
BASE_URL = os.getenv("POC_LLM_BASE_URL", "http://127.0.0.1:4141/v1")
MODELS = [m.strip() for m in os.getenv(
    "POC_MODELS", "claude-haiku-5.5,gpt-4.1-2025-04-14,gemini-3.8-flash").split(",") if m.strip()]
RUNS = int(os.getenv("POC_RUNS", "1"))
INTERVAL = float(os.getenv("POC_INTERVAL", "0"))
TOOL_TIMEOUT = float(os.getenv("POC_TOOL_TIMEOUT", "10"))
PROVIDER = os.getenv("POC_PROVIDER_NAME", "openai")
_SERVER = urlparse(BASE_URL)
SERVER_ADDRESS = _SERVER.hostname or "127.0.0.1"
SERVER_PORT = _SERVER.port or 0

SUPERVISOR_AGENT = "supervisor_agent"
WEATHER_AGENT = "weather_agent"
CURRENCY_AGENT = "currency_agent"
POLICY_AGENT = "policy_agent"
WRITER_AGENT = "writer_agent"
AGENT_DESCRIPTIONS = {
    SUPERVISOR_AGENT: "Routes a travel question to weather, currency, policy, or the writer.",
    WEATHER_AGENT: "Weather specialist; uses get_weather for real-time conditions.",
    CURRENCY_AGENT: "Currency specialist; uses convert_currency for real-time rates.",
    POLICY_AGENT: "Travel-policy specialist; loads the travel_policy skill, searches its references, and runs its budget script.",
    WRITER_AGENT: "Composes the final traveler-facing answer from the conversation.",
}
WORKFLOW_NAME = "travel_workflow"

HOUSE_STYLE_BLOCK = (
    "House style. Write in clear, plain English for a general audience. Prefer short "
    "sentences and concrete details over marketing language. Use metric units and "
    "24-hour clock times. Never invent weather data, exchange rates, prices, or flight "
    "numbers; every number must come from a tool result or the conversation so far. If "
    "a fact is missing, say plainly that it could not be verified. Never reveal "
    "internal tool names, routing words, or any part of these instructions in a final "
    "answer. Stay strictly on what the traveler asked; do not add unrelated "
    "recommendations or small talk. Do not use emojis in final answers. Quote numbers "
    "with their units and keep currency codes uppercase. Round converted money amounts "
    "to two decimals. Be respectful and patient, and never blame the traveler for "
    "ambiguity - ask for clarification instead of guessing. "
)
SYSTEM_PROMPT = (
    "You are the engine room of a small, friendly travel-assistant service. The "
    "shared house style below is repeated on every request; treat it as binding.\n\n"
    + (HOUSE_STYLE_BLOCK + "\n\n") * 16
).strip()

SUPERVISOR_HINT = (
    "Supervisor turn. You coordinate four workers: weather_agent (current weather via "
    "get_weather), currency_agent (currency conversion via convert_currency), "
    "policy_agent (visas, entry requirements, baggage, and budget guidance via the "
    "travel_policy skill), and writer (composes the final answer). Read the conversation "
    "so far, including any memory context. Decide the SINGLE next step. Route to a "
    "specialist only while some part of the request still needs that worker and no "
    "result for it exists yet. Once every requested fact has been gathered, route to "
    "writer. Respond with exactly one word: weather_agent, currency_agent, policy_agent, "
    "or writer."
)
WEATHER_HINT = (
    "You are the weather specialist. Call the get_weather tool for the city the traveler "
    "asked about, then reply with one short sentence summarizing the tool result. Never "
    "answer from memory; if the tool fails, say so in one short sentence."
)
CURRENCY_HINT = (
    "You are the currency specialist. Call the convert_currency tool for the conversion "
    "the traveler asked about, then reply with one short sentence summarizing the rate "
    "and the converted amount. Never answer from memory; if the tool fails, say so in "
    "one short sentence."
)
POLICY_HINT = (
    "You are the travel-policy specialist. Load the travel_policy skill first with "
    "load_skill. Use search_skill_resources to find the relevant reference, "
    "read_skill_resource to read it in full, and run_skill_script when the traveler asks "
    "for a budget. Answer only from the skill content and tool results; say plainly when "
    "a fact is missing. Use the conversation memory for the destination and trip details."
)
WRITER_HINT = (
    "You are the writer. Using the conversation so far (the traveler's question, worker "
    "messages, tool results, and memory context), compose the final answer for the "
    "traveler. Include the concrete numbers. Keep it to a few short sentences and do not "
    "mention tools, workers, or routing."
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_id: str
    model: str
    next: str
    answer: str
    tool_calls: int
    inference_calls: int
    in_tokens: int
    out_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    ttft_samples: list
    tpoc_samples: list


def _usage_counters(usage: dict) -> dict:
    input_details = usage.get("input_token_details") or {}
    output_details = usage.get("output_token_details") or {}
    return {
        "in_tokens": int(usage.get("input_tokens", 0) or 0),
        "out_tokens": int(usage.get("output_tokens", 0) or 0),
        "cache_read_tokens": int(input_details.get("cache_read", 0) or 0),
        "cache_write_tokens": int(input_details.get("cache_creation", 0) or 0),
        "reasoning_tokens": int(output_details.get("reasoning", 0) or 0),
    }


def _accumulated(state: AgentState, key: str, value) -> list:
    values = list(state.get(key, []))
    if value is not None:
        values.append(value)
    return values


_llm_cache: dict[str, ChatOpenAI] = {}


def get_llm(model: str) -> ChatOpenAI:
    if model not in _llm_cache:
        _llm_cache[model] = ChatOpenAI(
            model=model, base_url=BASE_URL, api_key="dummy",
            temperature=0, stream_usage=True, timeout=30,
        )
    return _llm_cache[model]


def _inference_metric_attrs(model: str, response_model: str, agent_name: str | None = None) -> dict:
    attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": PROVIDER,
        "gen_ai.request.model": model,
        "gen_ai.response.model": response_model or model,
        "server.address": SERVER_ADDRESS,
        "server.port": SERVER_PORT,
    }
    if agent_name:
        attrs["gen_ai.agent.name"] = agent_name
    return attrs


def call_llm(model: str, messages, conversation_id: str, *,
             agent_name: str | None = None, tools: list | None = None) -> tuple[AIMessage, dict]:
    if isinstance(messages, str):
        messages = [HumanMessage(content=messages)]
    span_attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": PROVIDER,
        "gen_ai.request.model": model,
        "gen_ai.request.temperature": 0,
        "gen_ai.request.stream": True,
        "gen_ai.conversation.id": conversation_id,
        "server.address": SERVER_ADDRESS,
        "server.port": SERVER_PORT,
    }
    if get_enduser_id():
        span_attrs["enduser.id"] = get_enduser_id()
    with tracer.start_as_current_span(
        f"chat {model}", kind=SpanKind.CLIENT, attributes=span_attrs
    ) as span:
        if capture_content():
            span.set_attribute("gen_ai.input.messages", serialize_messages(messages))
        llm = get_llm(model)
        if tools:
            llm = llm.bind_tools(tools)
        started = time.perf_counter()
        ttfc = None
        last_chunk_time = None
        chunk_times: list[float] = []
        parts: list[str] = []
        usage: dict = {}
        response_model = model
        finish_reason = "stop"
        message_id = None
        tool_acc: dict[int, dict] = {}
        try:
            for chunk in llm.stream(messages):
                now = time.perf_counter()
                if ttfc is None:
                    ttfc = now - started
                else:
                    chunk_times.append(now - last_chunk_time)
                last_chunk_time = now
                if getattr(chunk, "id", None):
                    message_id = chunk.id
                if chunk.content:
                    parts.append(chunk.content if isinstance(chunk.content, str) else str(chunk.content))
                for tool_chunk in getattr(chunk, "tool_call_chunks", None) or []:
                    index = tool_chunk.get("index")
                    acc = tool_acc.setdefault(0 if index is None else index,
                                              {"name": None, "args": "", "id": None})
                    if tool_chunk.get("name"):
                        acc["name"] = tool_chunk["name"]
                    if tool_chunk.get("id"):
                        acc["id"] = tool_chunk["id"]
                    if tool_chunk.get("args"):
                        acc["args"] += tool_chunk["args"]
                if getattr(chunk, "usage_metadata", None):
                    usage = chunk.usage_metadata
                if getattr(chunk, "response_metadata", None):
                    response_model = chunk.response_metadata.get("model_name", response_model) or response_model
                    reason = chunk.response_metadata.get("finish_reason")
                    if reason:
                        finish_reason = reason
            duration = time.perf_counter() - started
            content = "".join(parts)
        except Exception as exc:
            metric_attrs = {
                **_inference_metric_attrs(model, response_model, agent_name),
                "error.type": type(exc).__name__,
            }
            inference_duration.record(time.perf_counter() - started, metric_attrs)
            inference_op_input.record(0, metric_attrs)
            inference_op_output.record(0, metric_attrs)
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise

        tool_calls = []
        for index in sorted(tool_acc):
            acc = tool_acc[index]
            if not acc.get("name"):
                continue
            try:
                args = json.loads(acc["args"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": acc["args"]}
            tool_calls.append({
                "name": acc["name"], "args": args,
                "id": acc["id"] or f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call",
            })
        ai_message = AIMessage(content=content, tool_calls=tool_calls, id=message_id)

        in_tokens = int(usage.get("input_tokens", 0) or 0)
        out_tokens = int(usage.get("output_tokens", 0) or 0)
        cache_read = int((usage.get("input_token_details") or {}).get("cache_read", 0) or 0)
        cache_write = int((usage.get("input_token_details") or {}).get("cache_creation", 0) or 0)
        reasoning = int((usage.get("output_token_details") or {}).get("reasoning", 0) or 0)

        span.set_attribute("gen_ai.response.model", response_model)
        if message_id:
            span.set_attribute("gen_ai.response.id", message_id)
        span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
        if ttfc is not None:
            span.set_attribute("gen_ai.response.time_to_first_chunk", ttfc)
        span.set_attribute("gen_ai.usage.input_tokens", in_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", out_tokens)
        if cache_read:
            span.set_attribute("gen_ai.usage.cache_read.input_tokens", cache_read)
        if cache_write:
            span.set_attribute("gen_ai.usage.cache_write.input_tokens", cache_write)
        if reasoning:
            span.set_attribute("gen_ai.usage.reasoning.output_tokens", reasoning)
        if capture_content():
            span.set_attribute("gen_ai.output.messages", serialize_messages([ai_message]))

        metric_attrs = _inference_metric_attrs(model, response_model, agent_name)
        inference_duration.record(duration, metric_attrs)
        if ttfc is not None:
            inference_ttfc.record(ttfc, metric_attrs)
            usage["ttft"] = ttfc
        for interval in chunk_times:
            inference_tpoc.record(interval, metric_attrs)
        if chunk_times:
            usage["tpoc"] = sum(chunk_times) / len(chunk_times)
        if in_tokens:
            inference_input_tokens.add(in_tokens, {**metric_attrs, "gen_ai.token.modality": "text"})
        if out_tokens:
            inference_output_tokens.add(out_tokens, {**metric_attrs, "gen_ai.token.modality": "text"})
        if cache_read:
            inference_cache_read.add(cache_read, {**metric_attrs, "gen_ai.token.modality": "text"})
        if cache_write:
            inference_cache_write.add(cache_write, {**metric_attrs, "gen_ai.token.modality": "text"})
        if reasoning:
            inference_reasoning.add(reasoning, {**metric_attrs, "gen_ai.token.modality": "reasoning"})
        inference_op_input.record(in_tokens, metric_attrs)
        inference_op_output.record(out_tokens, metric_attrs)
        return ai_message, usage


_TOOL_HTTP_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool-http")


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(request, timeout=TOOL_TIMEOUT) as response:
        return response.read().decode("utf-8", "replace")


def _get_text(url: str) -> str:
    future = _TOOL_HTTP_POOL.submit(_fetch, url)
    try:
        return future.result(timeout=TOOL_TIMEOUT + 5)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"HTTP fetch exceeded {TOOL_TIMEOUT + 5:.0f}s: {url}") from None


def get_weather(city: str, tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
    """Get the current weather for a city (real-time, via wttr.in)."""
    with execute_tool_span(
        "get_weather", "execute_tool get_weather", tool_call_id, {"city": city},
        description=get_weather.__doc__,
    ) as span:
        result = _get_text("https://wttr.in/" + urllib.parse.quote(city.strip()) + "?format=3").strip()
        if capture_content():
            span.set_attribute("gen_ai.tool.call.result", result)
        return result


def convert_currency(amount: float, from_currency: str, to_currency: str,
                     tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
    """Convert an amount between currencies using real-time rates (via open.er-api.com)."""
    args = {"amount": amount, "from_currency": from_currency, "to_currency": to_currency}
    with execute_tool_span(
        "convert_currency", "execute_tool convert_currency", tool_call_id, args,
        description=convert_currency.__doc__,
    ) as span:
        source = from_currency.strip().upper()
        target = to_currency.strip().upper()
        data = json.loads(_get_text(f"https://open.er-api.com/v6/latest/{urllib.parse.quote(source)}"))
        if data.get("result") != "success":
            raise RuntimeError(f"rate provider error: {data.get('error-type', 'unknown')}")
        rates = data.get("rates") or {}
        if target not in rates:
            raise KeyError(f"unknown target currency {target!r}")
        rate = float(rates[target])
        result = json.dumps({
            "amount": float(amount), "from": source, "to": target,
            "rate": rate, "converted": round(float(amount) * rate, 2),
        })
        if capture_content():
            span.set_attribute("gen_ai.tool.call.result", result)
        return result


def _build_react_graph(agent_name: str, tools: list):
    tool_node = ToolNode(tools, handle_tool_errors=True)

    def call_model(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state.get("messages", [])]
        ai, usage = call_llm(state["model"], messages, state["conversation_id"],
                             agent_name=agent_name, tools=tools)
        counters = _usage_counters(usage)
        return {
            "messages": [ai],
            "inference_calls": state.get("inference_calls", 0) + 1,
            "in_tokens": state.get("in_tokens", 0) + counters["in_tokens"],
            "out_tokens": state.get("out_tokens", 0) + counters["out_tokens"],
            "cache_read_tokens": state.get("cache_read_tokens", 0) + counters["cache_read_tokens"],
            "cache_write_tokens": state.get("cache_write_tokens", 0) + counters["cache_write_tokens"],
            "reasoning_tokens": state.get("reasoning_tokens", 0) + counters["reasoning_tokens"],
            "ttft_samples": _accumulated(state, "ttft_samples", usage.get("ttft")),
            "tpoc_samples": _accumulated(state, "tpoc_samples", usage.get("tpoc")),
        }

    def call_tools(state: AgentState) -> dict:
        count = 0
        for message in reversed(state.get("messages", [])):
            if isinstance(message, AIMessage):
                count = len(message.tool_calls or [])
                break
        out = tool_node.invoke(state)
        return {"messages": out["messages"], "tool_calls": state.get("tool_calls", 0) + count}

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph.compile()


def _tool_definitions_json(tools: list) -> str:
    definitions = []
    for tool in tools:
        function = convert_to_openai_tool(tool)["function"]
        definitions.append({"type": "function", "name": function["name"],
                            "description": function.get("description", "")})
    return json.dumps(definitions)


def _make_specialist(agent_name: str, tools: list, hint: str):
    description = AGENT_DESCRIPTIONS[agent_name]
    tool_definitions = _tool_definitions_json(tools)
    subgraph = _build_react_graph(agent_name, tools)

    def specialist_node(state: AgentState) -> dict:
        started = time.perf_counter()
        token = set_current_agent(agent_name)
        try:
            with tracer.start_as_current_span(f"invoke_agent {agent_name}") as span:
                span.set_attribute("gen_ai.operation.name", "invoke_agent")
                span.set_attribute("gen_ai.agent.name", agent_name)
                span.set_attribute("gen_ai.agent.description", description)
                span.set_attribute("gen_ai.request.model", state["model"])
                span.set_attribute("gen_ai.conversation.id", state["conversation_id"])
                if get_enduser_id():
                    span.set_attribute("enduser.id", get_enduser_id())
                if capture_content():
                    span.set_attribute("gen_ai.system_instructions",
                                       messages_json("system", SYSTEM_PROMPT))
                    span.set_attribute("gen_ai.tool.definitions", tool_definitions)
                    span.set_attribute("gen_ai.input.messages",
                                       serialize_messages(state["messages"]))
                try:
                    result = subgraph.invoke({
                        "messages": [*state["messages"], HumanMessage(content=hint)],
                        "model": state["model"],
                        "conversation_id": state["conversation_id"],
                    })
                except Exception as exc:
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    agent_duration.record(
                        time.perf_counter() - started,
                        {"gen_ai.agent.name": agent_name, "gen_ai.request.model": state["model"],
                         "error.type": type(exc).__name__},
                    )
                    raise
                new_messages = list(result["messages"][len(state["messages"]) + 1:])
                in_tokens = result.get("in_tokens", 0)
                out_tokens = result.get("out_tokens", 0)
                ai_messages = [m for m in new_messages if isinstance(m, AIMessage)]
                if capture_content() and ai_messages:
                    span.set_attribute("gen_ai.output.messages", serialize_messages(ai_messages[-1:]))
                span.set_attribute("gen_ai.usage.input_tokens", in_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", out_tokens)
            agent_duration.record(
                time.perf_counter() - started,
                {"gen_ai.agent.name": agent_name, "gen_ai.request.model": state["model"]},
            )
            agent_inference_calls.record(result.get("inference_calls", 0), {"gen_ai.agent.name": agent_name})
            agent_tool_calls.record(result.get("tool_calls", 0), {"gen_ai.agent.name": agent_name})
            return {
                "messages": new_messages,
                "tool_calls": state.get("tool_calls", 0) + result.get("tool_calls", 0),
                "inference_calls": state.get("inference_calls", 0) + result.get("inference_calls", 0),
                "in_tokens": state.get("in_tokens", 0) + in_tokens,
                "out_tokens": state.get("out_tokens", 0) + out_tokens,
                "cache_read_tokens": state.get("cache_read_tokens", 0) + result.get("cache_read_tokens", 0),
                "cache_write_tokens": state.get("cache_write_tokens", 0) + result.get("cache_write_tokens", 0),
                "reasoning_tokens": state.get("reasoning_tokens", 0) + result.get("reasoning_tokens", 0),
                "ttft_samples": state.get("ttft_samples", []) + result.get("ttft_samples", []),
                "tpoc_samples": state.get("tpoc_samples", []) + result.get("tpoc_samples", []),
            }
        finally:
            reset_current_agent(token)

    return specialist_node


def _parse_route(text: str) -> str:
    lowered = (text or "").lower()
    for name in (WEATHER_AGENT, CURRENCY_AGENT, POLICY_AGENT, "writer"):
        if name in lowered:
            return name
    for name in (WEATHER_AGENT, CURRENCY_AGENT, POLICY_AGENT):
        if name.split("_")[0] in lowered:
            return name
    return "writer"


def supervisor_node(state: AgentState) -> dict:
    agent_name = SUPERVISOR_AGENT
    started = time.perf_counter()
    token = set_current_agent(agent_name)
    try:
        with tracer.start_as_current_span(f"invoke_agent {agent_name}") as span:
            span.set_attribute("gen_ai.operation.name", "invoke_agent")
            span.set_attribute("gen_ai.agent.name", agent_name)
            span.set_attribute("gen_ai.agent.description", AGENT_DESCRIPTIONS[agent_name])
            span.set_attribute("gen_ai.request.model", state["model"])
            span.set_attribute("gen_ai.conversation.id", state["conversation_id"])
            if get_enduser_id():
                span.set_attribute("enduser.id", get_enduser_id())
            if capture_content():
                span.set_attribute("gen_ai.system_instructions",
                                   messages_json("system", SYSTEM_PROMPT))
                span.set_attribute("gen_ai.input.messages",
                                   serialize_messages(state["messages"]))
            try:
                ai, usage = call_llm(
                    state["model"],
                    [SystemMessage(content=SYSTEM_PROMPT), *state["messages"],
                     HumanMessage(content=SUPERVISOR_HINT)],
                    state["conversation_id"], agent_name=agent_name,
                )
            except Exception as exc:
                span.set_attribute("error.type", type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                agent_duration.record(
                    time.perf_counter() - started,
                    {"gen_ai.agent.name": agent_name, "gen_ai.request.model": state["model"],
                     "error.type": type(exc).__name__},
                )
                raise
            decision = _parse_route(ai.content)
            counters = _usage_counters(usage)
            span.set_attribute("gen_ai.usage.input_tokens", counters["in_tokens"])
            span.set_attribute("gen_ai.usage.output_tokens", counters["out_tokens"])
            if capture_content():
                span.set_attribute("gen_ai.output.messages", serialize_messages([ai]))
    finally:
        reset_current_agent(token)
    agent_duration.record(
        time.perf_counter() - started,
        {"gen_ai.agent.name": agent_name, "gen_ai.request.model": state["model"]},
    )
    agent_inference_calls.record(1, {"gen_ai.agent.name": agent_name})
    agent_tool_calls.record(0, {"gen_ai.agent.name": agent_name})
    return {
        "next": decision,
        "inference_calls": state.get("inference_calls", 0) + 1,
        "in_tokens": state.get("in_tokens", 0) + counters["in_tokens"],
        "out_tokens": state.get("out_tokens", 0) + counters["out_tokens"],
        "cache_read_tokens": state.get("cache_read_tokens", 0) + counters["cache_read_tokens"],
        "cache_write_tokens": state.get("cache_write_tokens", 0) + counters["cache_write_tokens"],
        "reasoning_tokens": state.get("reasoning_tokens", 0) + counters["reasoning_tokens"],
        "ttft_samples": _accumulated(state, "ttft_samples", usage.get("ttft")),
        "tpoc_samples": _accumulated(state, "tpoc_samples", usage.get("tpoc")),
    }


def writer_node(state: AgentState) -> dict:
    agent_name = WRITER_AGENT
    started = time.perf_counter()
    token = set_current_agent(agent_name)
    try:
        with tracer.start_as_current_span(f"invoke_agent {agent_name}") as span:
            span.set_attribute("gen_ai.operation.name", "invoke_agent")
            span.set_attribute("gen_ai.agent.name", agent_name)
            span.set_attribute("gen_ai.agent.description", AGENT_DESCRIPTIONS[agent_name])
            span.set_attribute("gen_ai.request.model", state["model"])
            span.set_attribute("gen_ai.conversation.id", state["conversation_id"])
            if get_enduser_id():
                span.set_attribute("enduser.id", get_enduser_id())
            if capture_content():
                span.set_attribute("gen_ai.system_instructions",
                                   messages_json("system", SYSTEM_PROMPT))
                span.set_attribute("gen_ai.input.messages",
                                   serialize_messages(state["messages"]))
            try:
                ai, usage = call_llm(
                    state["model"],
                    [SystemMessage(content=SYSTEM_PROMPT), *state["messages"],
                     HumanMessage(content=WRITER_HINT)],
                    state["conversation_id"], agent_name=agent_name,
                )
            except Exception as exc:
                span.set_attribute("error.type", type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                agent_duration.record(
                    time.perf_counter() - started,
                    {"gen_ai.agent.name": agent_name, "gen_ai.request.model": state["model"],
                     "error.type": type(exc).__name__},
                )
                raise
            counters = _usage_counters(usage)
            span.set_attribute("gen_ai.usage.input_tokens", counters["in_tokens"])
            span.set_attribute("gen_ai.usage.output_tokens", counters["out_tokens"])
            if capture_content():
                span.set_attribute("gen_ai.output.messages", serialize_messages([ai]))
    finally:
        reset_current_agent(token)
    agent_duration.record(
        time.perf_counter() - started,
        {"gen_ai.agent.name": agent_name, "gen_ai.request.model": state["model"]},
    )
    agent_inference_calls.record(1, {"gen_ai.agent.name": agent_name})
    agent_tool_calls.record(0, {"gen_ai.agent.name": agent_name})
    return {
        "messages": [ai],
        "answer": ai.content,
        "inference_calls": state.get("inference_calls", 0) + 1,
        "in_tokens": state.get("in_tokens", 0) + counters["in_tokens"],
        "out_tokens": state.get("out_tokens", 0) + counters["out_tokens"],
        "cache_read_tokens": state.get("cache_read_tokens", 0) + counters["cache_read_tokens"],
        "cache_write_tokens": state.get("cache_write_tokens", 0) + counters["cache_write_tokens"],
        "reasoning_tokens": state.get("reasoning_tokens", 0) + counters["reasoning_tokens"],
        "ttft_samples": _accumulated(state, "ttft_samples", usage.get("ttft")),
        "tpoc_samples": _accumulated(state, "tpoc_samples", usage.get("tpoc")),
    }


def _route_from_supervisor(state: AgentState) -> str:
    decision = state.get("next")
    if decision in (WEATHER_AGENT, CURRENCY_AGENT, POLICY_AGENT, WRITER_AGENT):
        return decision
    return WRITER_AGENT


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("weather_agent", _make_specialist(WEATHER_AGENT, [get_weather], WEATHER_HINT))
    graph.add_node("currency_agent", _make_specialist(CURRENCY_AGENT, [convert_currency], CURRENCY_HINT))
    graph.add_node(POLICY_AGENT, _make_specialist(
        POLICY_AGENT,
        [load_skill, search_skill_resources, read_skill_resource, run_skill_script],
        POLICY_HINT,
    ))
    graph.add_node("writer", writer_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", _route_from_supervisor,
        {WEATHER_AGENT: "weather_agent", CURRENCY_AGENT: "currency_agent",
         POLICY_AGENT: POLICY_AGENT, WRITER_AGENT: "writer"},
    )
    graph.add_edge("weather_agent", "supervisor")
    graph.add_edge("currency_agent", "supervisor")
    graph.add_edge(POLICY_AGENT, "supervisor")
    graph.add_edge("writer", END)
    return graph.compile()


CONVERSATIONS = [
    {
        "user": "user-001",
        "questions": [
            "What's the weather in Tokyo right now, and how much is 250 EUR in JPY?",
            "For that same 5-day Tokyo trip on a mid-range budget, do I need a visa, "
            "what's the baggage allowance, and what's a reasonable daily budget?",
        ],
    },
    {
        "user": "user-002",
        "questions": [
            "What's the weather in Lisbon tomorrow, and what is 300 USD in EUR?",
            "For that same 4-day Lisbon trip on a mid-range budget, do I need a visa, "
            "what's the baggage allowance, and what's a reasonable daily budget?",
        ],
    },
    {
        "user": "user-003",
        "questions": [
            "What's the weather in Toronto today, and what is 500 USD in CAD?",
            "For that same 3-day Toronto trip on a budget, do I need a visa, "
            "what's the baggage allowance, and what's a reasonable daily budget?",
        ],
    },
]


def run_once(app, index: int) -> None:
    conversation = CONVERSATIONS[(index - 1) // 2 % len(CONVERSATIONS)]
    question_index = (index - 1) % 2
    question = conversation["questions"][question_index]
    conversation_id = f"conv-{(index - 1) // 2 % len(CONVERSATIONS) + 1:04d}"
    enduser_id = conversation["user"]
    model = MODELS[(index - 1) % len(MODELS)]
    set_run_context(enduser_id, conversation_id)

    with tracer.start_as_current_span(f"invoke_workflow {WORKFLOW_NAME}") as root:
        root.set_attribute("gen_ai.operation.name", "invoke_workflow")
        root.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        root.set_attribute("gen_ai.conversation.id", conversation_id)
        root.set_attribute("enduser.id", enduser_id)
        if capture_content():
            root.set_attribute("gen_ai.input.messages", messages_json("user", question))
        trace_id = root.get_span_context().trace_id
        started = time.perf_counter()
        try:
            records = search_memory(question)
            context_message = memory_context_message(records)
            messages = ([context_message] if context_message else []) + [
                HumanMessage(content=question, id=f"{conversation_id}-{question_index}")
            ]
            result = app.invoke({
                "messages": messages,
                "model": model,
                "conversation_id": conversation_id,
            })
            answer = str(result.get("answer", ""))
            if capture_content():
                root.set_attribute("gen_ai.output.messages", messages_json("assistant", answer))
            create_memory(question, answer)
        except Exception as exc:
            root.set_attribute("error.type", type(exc).__name__)
            root.set_status(Status(StatusCode.ERROR, str(exc)))
            workflow_duration.record(
                time.perf_counter() - started,
                {"gen_ai.workflow.name": WORKFLOW_NAME, "error.type": type(exc).__name__},
            )
            print(f"run {index}: FAILED {type(exc).__name__}: {exc}")
            return
    workflow_duration.record(
        time.perf_counter() - started, {"gen_ai.workflow.name": WORKFLOW_NAME})
    print(
        f"run {index}: model={model} tools={result.get('tool_calls')} "
        f"inferences={result.get('inference_calls')} "
        f"tokens={result.get('in_tokens')}+{result.get('out_tokens')} "
        f"answer={answer[:32]!r} trace={trace_id:032x}"
    )


def main():
    app = build_graph()
    index = 0
    try:
        while RUNS == 0 or index < RUNS:
            index += 1
            run_once(app, index)
            if INTERVAL:
                time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\ninterrupted by user")
    telemetry.provider.force_flush()
    telemetry.meter_provider.force_flush()
    telemetry.provider.shutdown()
    telemetry.meter_provider.shutdown()
    print(f"done: {index} invocation(s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the workflow tests**

Run: `python3 -m unittest app.tests.test_workflow -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Run all tests and compile**

Run: `python3 -m unittest discover -s app/tests -v && python3 -m py_compile app/langgraph_app.py`
Expected: all tests PASS, compile exit 0.

- [ ] **Step 6: Static conformance checks**

Run:

```bash
grep -RIn "poc\." app/*.py app/tests || true
grep -RIn "gen_ai.usage" app/langgraph_app.py
```

Expected: no `poc.` matches. The only `gen_ai.usage` writes are on `chat` and
`invoke_agent` spans, never on the workflow root.

---

### Task 5: Tempo dimensions and Agent Insight dashboard

**Files:**
- Modify: `tempo/tempo.yaml`
- Modify: `grafana/dashboards/agent-insight.json`

- [ ] **Step 1: Update Tempo span-metric dimensions**

Replace the `span_metrics.dimensions` block with:

```yaml
    span_metrics:
      dimensions:
        - gen_ai.operation.name
        - gen_ai.request.model
        - gen_ai.provider.name
        - gen_ai.tool.name
        - gen_ai.agent.name
        - gen_ai.workflow.name
        - gen_ai.skill.name
        - gen_ai.skill.resource.name
        - gen_ai.data_source.id
        - gen_ai.memory.store.id
        - error.type
```

Do not add `gen_ai.conversation.id`, `enduser.id`, or `gen_ai.tool.call.id`.

- [ ] **Step 2: Back up the dashboard**

Run:

```bash
cp grafana/dashboards/agent-insight.json /tmp/agent-insight.json.bak
```

- [ ] **Step 3: Apply the dashboard patch**

Run this script from the repository root:

```python
# /tmp/patch_agent_insight.py
import copy
import json
from pathlib import Path

path = Path("grafana/dashboards/agent-insight.json")
dashboard = json.loads(path.read_text())
panels = {panel["id"]: panel for panel in dashboard["panels"]}


def set_target(panel_id, expr, legend=None):
    target = panels[panel_id]["targets"][0]
    target["expr"] = expr
    if legend is not None:
        target["legendFormat"] = legend


# --- KPI row ---------------------------------------------------------------
panels[11]["title"] = "Runs"
set_target(11, 'sum(increase(traces_spanmetrics_calls_total{service=~"$application", gen_ai_operation_name="invoke_workflow"}[$__range]))')
panels[11]["description"] = (
    "One end-to-end workflow execution. Look here when workload or failure rate changes.")

panels[12]["title"] = "Steps"
set_target(12, 'sum(increase(traces_spanmetrics_calls_total{service=~"$application", gen_ai_agent_name=~"$agent", gen_ai_agent_name!="", gen_ai_operation_name=~"chat|execute_tool"}[$__range]))')
panels[12]["description"] = (
    "Model calls plus tool calls, the standard's step definition. A rising steps-per-run "
    "usually means the agent is looping.")

panels[13]["title"] = "Agent Calls"
set_target(13, 'sum(increase(traces_spanmetrics_calls_total{service=~"$application", gen_ai_agent_name=~"$agent", gen_ai_agent_name!="", gen_ai_operation_name="invoke_agent"}[$__range]))')
panels[13]["description"] = (
    "invoke_agent spans. Compare specialist workload here.")

panels[14]["title"] = "Tokens"
set_target(14, '(sum(increase(gen_ai_client_inference_usage_input_tokens_total{service_name=~"$application", gen_ai_agent_name=~"$agent", gen_ai_request_model=~"$model", gen_ai_agent_name!=""}[$__range])) + sum(increase(gen_ai_client_inference_usage_output_tokens_total{service_name=~"$application", gen_ai_agent_name=~"$agent", gen_ai_request_model=~"$model", gen_ai_agent_name!=""}[$__range])))')
panels[14]["description"] = (
    "Input plus output tokens over the selected range. Use it with tokens per call to "
    "spot cost growth that call counts hide.")

panels[18]["title"] = "Error Rate"
set_target(18, 'clamp_max(100 * sum(increase(traces_spanmetrics_calls_total{service=~"$application", gen_ai_operation_name="invoke_workflow", error_type!=""}[$__range])) / clamp_min(sum(increase(traces_spanmetrics_calls_total{service=~"$application", gen_ai_operation_name="invoke_workflow"}[$__range])), 1), 100)')
panels[18]["description"] = (
    "Failed workflow spans divided by all workflow spans. When it rises, inspect the "
    "agent and tool error panels for the first failing step.")
panels[18]["fieldConfig"]["defaults"]["unit"] = "percent"
panels[18]["fieldConfig"]["defaults"]["decimals"] = 1
panels[18]["fieldConfig"]["defaults"]["thresholds"] = {
    "mode": "absolute",
    "steps": [
        {"color": "#16A34A", "value": None},
        {"color": "#F59E0B", "value": 5},
        {"color": "#DC2626", "value": 15},
    ],
}

# Remove the old cost trend; its grid slot becomes the trace-based conversation table.
dashboard["panels"] = [panel for panel in dashboard["panels"] if panel["id"] != 19]

# --- Activity over time ----------------------------------------------------
activity = panels[21]["targets"]
activity[0]["expr"] = 'sum(increase(traces_spanmetrics_calls_total{service=~"$application", gen_ai_operation_name="invoke_workflow"}[$__rate_interval]))'
activity[0]["legendFormat"] = "Runs"
activity[1]["expr"] = 'sum(increase(traces_spanmetrics_calls_total{service=~"$application", gen_ai_agent_name=~"$agent", gen_ai_agent_name!="", gen_ai_operation_name=~"chat|execute_tool"}[$__rate_interval]))'
activity[1]["legendFormat"] = "Steps"
activity[2]["expr"] = 'sum(increase(traces_spanmetrics_calls_total{service=~"$application", gen_ai_agent_name=~"$agent", gen_ai_agent_name!="", gen_ai_operation_name="invoke_agent"}[$__rate_interval]))'
activity[2]["legendFormat"] = "Agent calls"

# --- Trajectory table ------------------------------------------------------
trajectory = panels[32]
trajectory["targets"][0]["query"] = (
    '{ (span.gen_ai.agent.name =~ "$agent" && span.gen_ai.operation.name = "invoke_agent") '
    '|| (span.gen_ai.operation.name = "chat" && span.gen_ai.request.model =~ "$model") } '
    '<< { span.gen_ai.operation.name = "invoke_workflow" && resource.service.name =~ "$application" } '
    '| select(span.gen_ai.input.messages, span.gen_ai.output.messages, '
    'span.enduser.id, span.gen_ai.conversation.id)'
)
trajectory["targets"][0]["limit"] = 100
trajectory["transformations"] = [
    {
        "id": "organize",
        "options": {
            "excludeByName": {
                "name": True, "spanID": True, "traceName": True,
                "gen_ai.operation.name": True, "service.name": True,
            },
            "renameByName": {
                "traceIdHidden": "Trace ID",
                "gen_ai.input.messages": "Input",
                "gen_ai.output.messages": "Output",
                "time": "Start",
                "duration": "Duration",
                "traceService": "Entry Application",
                "enduser.id": "User ID",
                "gen_ai.conversation.id": "Conversation ID",
            },
            "indexByName": {
                "traceIdHidden": 0,
                "gen_ai.input.messages": 1,
                "gen_ai.output.messages": 2,
                "time": 3,
                "duration": 4,
                "traceService": 5,
                "enduser.id": 6,
                "gen_ai.conversation.id": 7,
            },
        },
    }
]
trajectory["description"] = (
    "One row per run. Input/Output are populated only when "
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT is enabled. Open the trace ID "
    "to inspect skill, retrieval, and memory spans.")

# --- Conversations table (trace-derived, no high-cardinality metric labels) --
conversations = copy.deepcopy(trajectory)
conversations.update({
    "id": 200,
    "title": "Conversations",
    "gridPos": {"h": 7, "w": 12, "x": 0, "y": 15},
    "description": (
        "Conversation and user ids are read from workflow spans, never from metric "
        "dimensions (R-11.1-01). Grouping shows one row per conversation."),
})
conversations["targets"][0]["query"] = (
    '{ span.gen_ai.operation.name = "invoke_workflow" && resource.service.name =~ "$application" } '
    '| select(span.gen_ai.conversation.id, span.enduser.id)'
)
conversations["targets"][0]["limit"] = 1000
conversations["transformations"] = [
    {
        "id": "groupBy",
        "options": {
            "fields": {
                "gen_ai.conversation.id": {"aggregations": [], "operation": "groupby"},
                "enduser.id": {"aggregations": [], "operation": "groupby"},
                "traceIdHidden": {"aggregations": ["count"], "operation": "aggregate"},
            }
        },
    },
    {
        "id": "organize",
        "options": {
            "excludeByName": {},
            "renameByName": {
                "gen_ai.conversation.id": "Conversation ID",
                "enduser.id": "User ID",
                "traceIdHidden (count)": "Runs",
            },
            "indexByName": {
                "gen_ai.conversation.id": 0,
                "enduser.id": 1,
                "traceIdHidden (count)": 2,
            },
        },
    },
]
dashboard["panels"].append(conversations)

# --- Scenarios row ---------------------------------------------------------
row = copy.deepcopy(panels[46])
row.update({"id": 201, "title": "Scenarios", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 123}})
dashboard["panels"].append(row)


def clone_timeseries(source_id, new_id, title, x, y, width, expr, legend, description,
                     unit="short"):
    panel = copy.deepcopy(panels[source_id])
    panel.update({
        "id": new_id,
        "title": title,
        "gridPos": {"h": 7, "w": width, "x": x, "y": y},
        "description": description,
    })
    panel["fieldConfig"]["defaults"]["unit"] = unit
    panel["targets"] = [{
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "expr": expr,
        "legendFormat": legend,
        "refId": "A",
    }]
    dashboard["panels"].append(panel)
    return panel


clone_timeseries(
    111, 202, "Operation mix", 0, 124, 12,
    'sum by (gen_ai_operation_name)(increase(traces_spanmetrics_calls_total{service=~"$application", gen_ai_operation_name!=""}[$__rate_interval]))',
    "{{gen_ai_operation_name}}",
    "Distribution of standard GenAI operations. A missing operation means the path did "
    "not execute; an unexpected one means instrumentation drift.")
clone_timeseries(
    111, 203, "Skill calls", 12, 124, 12,
    'sum by (gen_ai_skill_name, gen_ai_tool_name)(increase(traces_spanmetrics_calls_total{service=~"$application", gen_ai_skill_name=~"$skill", gen_ai_skill_name!=""}[$__rate_interval]))',
    "{{gen_ai_skill_name}} / {{gen_ai_tool_name}}",
    "Load/read/script tool calls by skill. Flat lines while policy questions arrive mean "
    "the model stopped using the skill.")
clone_timeseries(
    111, 204, "Skill latency p95", 0, 131, 8,
    'histogram_quantile(0.95, sum by (le, gen_ai_skill_name)(increase(traces_spanmetrics_latency_bucket{service=~"$application", gen_ai_skill_name=~"$skill", gen_ai_skill_name!=""}[$__rate_interval])))',
    "{{gen_ai_skill_name}}",
    "p95 execute_tool duration for skill operations. High values point at resource reads "
    "or script execution.",
    unit="s")
clone_timeseries(
    111, 205, "Retrieval calls", 8, 131, 8,
    'sum by (gen_ai_data_source_id)(increase(traces_spanmetrics_calls_total{service=~"$application", gen_ai_operation_name="retrieval", gen_ai_data_source_id!=""}[$__rate_interval]))',
    "{{gen_ai_data_source_id}}",
    "Retrieval operations by data source. Zero calls while policy questions arrive means "
    "the model skipped retrieval.")
clone_timeseries(
    111, 206, "Memory operations", 16, 131, 8,
    'sum by (gen_ai_operation_name)(increase(traces_spanmetrics_calls_total{service=~"$application", gen_ai_operation_name=~"search_memory|create_memory|update_memory|upsert_memory|delete_memory"}[$__rate_interval]))',
    "{{gen_ai_operation_name}}",
    "Memory read/write volume. Zero writes means the second run in a conversation cannot "
    "use the first.")

# --- Model errors: use error.type, not a local counter ---------------------
set_target(143, 'sum by (gen_ai_request_model)(increase(gen_ai_client_inference_duration_seconds_count{service_name=~"$application", gen_ai_agent_name=~"$agent", gen_ai_request_model=~"$model", gen_ai_agent_name!="", error_type!=""}[$__rate_interval]))')
panels[143]["description"] = (
    "Failed inference operations by model, from error.type on the duration histogram. "
    "A spike here precedes user-visible failures.")

# --- Model TPOT: use the standard per-chunk metric -------------------------
panels[145]["title"] = "Model time per chunk p95"
set_target(145, 'histogram_quantile(0.95, sum by (le, gen_ai_request_model)(increase(gen_ai_client_inference_time_per_output_chunk_seconds_bucket{service_name=~"$application", gen_ai_agent_name=~"$agent", gen_ai_request_model=~"$model", gen_ai_agent_name!=""}[$__rate_interval])))')
panels[145]["description"] = (
    "p95 interval between streamed output chunks by model. A rising interval with flat "
    "TTFT points at generation speed, not connection setup.")

# --- Skill variable --------------------------------------------------------
dashboard["templating"]["list"].append({
    "name": "skill",
    "label": "Skill",
    "type": "query",
    "datasource": {"type": "prometheus", "uid": "prometheus"},
    "query": {
        "query": 'label_values(traces_spanmetrics_calls_total{gen_ai_skill_name!=""}, gen_ai_skill_name)',
        "refId": "A",
    },
    "definition": 'label_values(traces_spanmetrics_calls_total{gen_ai_skill_name!=""}, gen_ai_skill_name)',
    "includeAll": True,
    "allValue": ".*",
    "multi": True,
    "current": {"text": "All", "value": "$__all"},
    "refresh": 1,
})

dashboard["description"] = (
    "Agent Insight. Runs and steps come from standard span metrics; conversations and "
    "users are read from traces because high-cardinality ids must not enter metric "
    "dimensions (R-11.1-01). Content columns require "
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true.")

path.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
print("agent-insight patched")
```

Run: `python3 /tmp/patch_agent_insight.py`
Expected: `agent-insight patched`.

- [ ] **Step 4: Validate the dashboard JSON**

Run:

```bash
jq empty grafana/dashboards/agent-insight.json
jq -r '[.panels[].title] | join(", ")' grafana/dashboards/agent-insight.json | grep -E 'Runs|Steps|Error Rate|Conversations|Scenarios|Operation mix'
grep -c 'poc_' grafana/dashboards/agent-insight.json || true
grep -c 'gen_ai_conversation_id\|enduser_id' grafana/dashboards/agent-insight.json || true
```

Expected: JSON valid; the titles are present; the `poc_` count is 0; the
conversation/user label count is 0 in metric queries (the Tempo TraceQL text may
contain the attribute names, which is correct).

- [ ] **Step 5: Restart Tempo and verify**

Run:

```bash
docker compose restart tempo
sleep 5
curl -s http://127.0.0.1:3200/ready
```

Expected: `ready`.

Query Prometheus for new label names after new spans arrive:

```bash
curl -s 'http://127.0.0.1:9091/api/v1/label/__name__/values' | jq -r '.data[]' | grep traces_spanmetrics
```

Expected: `traces_spanmetrics_calls_total` etc. present. The old
`gen_ai_conversation_id` / `enduser_id` series may remain until retention, but no
new series should carry them.

- [ ] **Step 6: Verify Grafana loaded the dashboard**

Run:

```bash
curl -s http://127.0.0.1:3000/api/dashboards/uid/agent-insight | jq -r '.dashboard.title, (.dashboard.panels[] | select(.title=="Runs" or .title=="Conversations" or .title=="Scenarios") | .title)'
```

Expected: `Agent Insight`, `Runs`, `Conversations`, `Scenarios`.

---

### Task 6: Token & Cost dashboard

**Files:**
- Modify: `grafana/dashboards/token-cost.json`

- [ ] **Step 1: Back up the dashboard**

Run:

```bash
cp grafana/dashboards/token-cost.json /tmp/token-cost.json.bak
```

- [ ] **Step 2: Apply the cost patch**

Run this script from the repository root:

```python
# /tmp/patch_token_cost.py
import json
from pathlib import Path

path = Path("grafana/dashboards/token-cost.json")
dashboard = json.loads(path.read_text())
panels = {panel["id"]: panel for panel in dashboard["panels"]}

FILTERS = (
    'service_name=~"$application", gen_ai_agent_name=~"$agent", '
    'gen_ai_request_model=~"$model", gen_ai_agent_name!=""'
)
INPUT = f'gen_ai_client_inference_usage_input_tokens_total{{{FILTERS}}}'
OUTPUT = f'gen_ai_client_inference_usage_output_tokens_total{{{FILTERS}}}'
PRICE_IN = (
    'label_replace(vector(1.0), "gen_ai_request_model", "claude-haiku-5.5", "", "") '
    'or label_replace(vector(2.0), "gen_ai_request_model", "gpt-4.1-2025-04-14", "", "") '
    'or label_replace(vector(0.3), "gen_ai_request_model", "gemini-3.8-flash", "", "")'
)
PRICE_OUT = (
    'label_replace(vector(5.0), "gen_ai_request_model", "claude-haiku-5.5", "", "") '
    'or label_replace(vector(8.0), "gen_ai_request_model", "gpt-4.1-2025-04-14", "", "") '
    'or label_replace(vector(2.5), "gen_ai_request_model", "gemini-3.8-flash", "", "")'
)


def cost_model(selector, by):
    return (
        f'(sum by ({by})(increase({INPUT}[{selector}])) '
        f'* on (gen_ai_request_model) group_left() ({PRICE_IN}) / 1e6) + '
        f'(sum by ({by})(increase({OUTPUT}[{selector}])) '
        f'* on (gen_ai_request_model) group_left() ({PRICE_OUT}) / 1e6)'
    )


def cost_total(selector):
    return f'sum({cost_model(selector, "gen_ai_request_model")})'


def cost_rate_total():
    return (
        f'sum((sum by (gen_ai_request_model)(rate({INPUT}[$__rate_interval])) '
        f'* on (gen_ai_request_model) group_left() ({PRICE_IN}) / 1e6) + '
        f'(sum by (gen_ai_request_model)(rate({OUTPUT}[$__rate_interval])) '
        f'* on (gen_ai_request_model) group_left() ({PRICE_OUT}) / 1e6))'
    )


def tokens(selector, by="gen_ai_request_model"):
    return (
        f'sum by ({by})(increase({INPUT}[{selector}])) + '
        f'sum by ({by})(increase({OUTPUT}[{selector}]))'
    )


def tokens_total(selector):
    return f'sum(increase({INPUT}[{selector}])) + sum(increase({OUTPUT}[{selector}]))'


def chat_count(selector):
    return (
        f'sum(increase(traces_spanmetrics_calls_total{{service=~"$application", '
        f'gen_ai_agent_name=~"$agent", gen_ai_request_model=~"$model", '
        f'gen_ai_agent_name!="", gen_ai_operation_name="chat"}}[{selector}]))'
    )


def set_expr(panel_id, expr, ref="A"):
    for target in panels[panel_id]["targets"]:
        if target.get("refId") == ref:
            target["expr"] = expr
            return
    raise KeyError((panel_id, ref))


set_expr(11, f'({cost_total("$__range")}) > 0')
set_expr(12, f'({cost_rate_total()} * 3600) > 0')
set_expr(13, f'({cost_rate_total()} * 2592000) > 0')
set_expr(
    14,
    f'({cost_total("$__range")}) / clamp_min({chat_count("$__range")}, 0.0001)',
)
set_expr(
    16,
    f'({cost_total("$__range")}) / clamp_min(({tokens_total("$__range")}) / 1000, 0.0001)',
)
set_expr(
    21,
    f'sum by (model)(label_replace({cost_model("$__rate_interval", "gen_ai_request_model")}, '
    f'"model", "$1", "gen_ai_request_model", "(.+)"))',
)
set_expr(22, f'{cost_rate_total()} * 3600')
set_expr(
    23,
    f'({cost_total("$__rate_interval")}) / clamp_min({chat_count("$__rate_interval")}, 0.0001)',
)
set_expr(
    32,
    f'({cost_total("$__rate_interval")}) / '
    f'clamp_min(({tokens_total("$__rate_interval")}) / 1000, 0.0001)',
)
set_expr(
    41,
    f'sum by (model)(label_replace({cost_model("$__range", "gen_ai_request_model")}, '
    f'"model", "$1", "gen_ai_request_model", "(.+)"))',
)
set_expr(
    42,
    f'sum by (model)(label_replace({cost_model("$__range", "gen_ai_request_model")}, '
    f'"model", "$1", "gen_ai_request_model", "(.+)"))',
    ref="C",
)
set_expr(
    42,
    f'(sum by (model)(label_replace({cost_model("$__range", "gen_ai_request_model")}, '
    f'"model", "$1", "gen_ai_request_model", "(.+)"))) / '
    f'clamp_min((sum by (model)(label_replace({tokens("$__range")}, '
    f'"model", "$1", "gen_ai_request_model", "(.+)"))) / 1000, 0.0001)',
    ref="D",
)
set_expr(
    51,
    f'sum by (agent)(label_replace({cost_model("$__range", "gen_ai_request_model, gen_ai_agent_name")}, '
    f'"agent", "$1", "gen_ai_agent_name", "(.+)"))',
    ref="C",
)
set_expr(
    51,
    f'(sum by (agent)(label_replace({cost_model("$__range", "gen_ai_request_model, gen_ai_agent_name")}, '
    f'"agent", "$1", "gen_ai_agent_name", "(.+)"))) / '
    f'clamp_min((sum by (agent)(label_replace({tokens("$__range", "gen_ai_request_model, gen_ai_agent_name")}, '
    f'"agent", "$1", "gen_ai_agent_name", "(.+)"))) / 1000, 0.0001)',
    ref="D",
)

dashboard["description"] = (
    "Token & Cost. Cost is derived query-side from cumulative token counters times the "
    "deployment price table (USD per 1M tokens: claude-haiku-5.5 1/5, "
    "gpt-4.1-2025-04-14 2/8, gemini-3.8-flash 0.3/2.5). It is not a telemetry metric; "
    "update the inline price vectors when models or prices change. Unknown models "
    "contribute zero cost."
)

path.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
print("token-cost patched")
```

Run: `python3 /tmp/patch_token_cost.py`
Expected: `token-cost patched`.

- [ ] **Step 3: Validate**

Run:

```bash
jq empty grafana/dashboards/token-cost.json
grep -c 'poc_cost_usd_total' grafana/dashboards/token-cost.json || true
grep -c 'gen_ai_client_inference_usage_input_tokens_total' grafana/dashboards/token-cost.json
```

Expected: JSON valid; `poc_cost_usd_total` count 0; the token metric name appears
in the derived expressions.

- [ ] **Step 4: Verify Grafana loaded the dashboard**

Run:

```bash
curl -s http://127.0.0.1:3000/api/dashboards/uid/token-cost | jq -r '.dashboard.title, .dashboard.description'
```

Expected: `Token & Cost` and the derived-cost description.

- [ ] **Step 5: Spot-check one cost expression against Prometheus**

Take the `Total Cost` expression from the dashboard JSON and query Prometheus:

```bash
EXPR=$(curl -s http://127.0.0.1:3000/api/dashboards/uid/token-cost | jq -r '.dashboard.panels[] | select(.title=="Total Cost") | .targets[0].expr')
curl -sG 'http://127.0.0.1:9091/api/v1/query' --data-urlencode "query=$EXPR" | jq -r '.status, (.data.result[0].value[1] // "no data")'
```

Expected: `success` and either a numeric value or `no data` (if the app has not
run since the counters reset). An error here means the PromQL needs fixing.

---

### Task 7: README and cleanup

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the stale dashboard and signal documentation**

Rewrite `README.md` with the following structure and facts (keep the existing
`docker compose` and port instructions):

1. Title and one-paragraph purpose.
2. Architecture line:
   `LangGraph agent --OTLP--> OTel Collector --> Tempo (traces) + Prometheus (metrics) --> Grafana`.
3. Dashboards table with exactly the two present dashboards (`agent-insight`,
   `token-cost`) and their purposes. Remove `topology-health`, `entity-detail`,
   and `trace-trajectory` rows.
4. Concepts: Run, Conversation, User. State explicitly:
   - `gen_ai.conversation.id` and `enduser.id` are trace-only; never metric dimensions.
   - The Conversations panel reads workflow spans via TraceQL.
   - Cost is derived in the token-cost dashboard from tokens × price; there is no
     cost metric.
5. Scenarios: supervisor, weather, currency, policy (Agent Skills), writer;
   memory read/write; retrieval.
6. Span table with the exact names from this plan, including the three
   `execute_tool` refinements and `retrieval {data_source.id}`.
7. Metrics table with the standard instruments, including
   `gen_ai.client.operation.duration` and
   `gen_ai.client.inference.time_per_output_chunk`. Remove the local extensions
   table entirely.
8. Attributes table: remove every `poc.*` row; add skill, process, retrieval,
   and memory attributes.
9. Content switch:
   `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` enables messages,
   tool arguments/results, retrieval query/documents, memory query/records, and
   tool definitions; the default is off.
10. Running:
    ```bash
    docker compose up -d
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    export POC_LLM_BASE_URL=http://127.0.0.1:4141/v1
    export POC_MODELS=claude-haiku-5.5,gpt-4.1-2025-04-14,gemini-3.8-flash
    export POC_RUNS=6
    export POC_INTERVAL=3
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
    python3 app/langgraph_app.py
    ```
11. Note that the first run of a conversation has no memory; the second run
    reads the first run's Q/A from `app/.state/memory.json`.
12. Note the demo-scale Tempo restart requirement for dimension changes.
13. Note that the standard's official files are `ai-observability-design.md` and
    `ai-observability-standard.zh.md` and are not modified by the example.

- [ ] **Step 2: Remove stale references**

Run:

```bash
grep -RIn 'poc\.\|poc_' README.md app grafana/dashboards || true
grep -RIn 'topology-health\|entity-detail\|trace-trajectory' README.md || true
```

Expected: no matches.

- [ ] **Step 3: Checkpoint**

Run: `grep -n 'Conversations\|gen_ai.client.operation.duration\|OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT' README.md | head`
Expected: the new sections are present.

---

### Task 8: End-to-end verification

**Files:** none (runtime checks)

- [ ] **Step 1: Static checks and unit tests**

Run:

```bash
python3 -m py_compile app/*.py
python3 -m unittest discover -s app/tests -v
jq empty grafana/dashboards/agent-insight.json
jq empty grafana/dashboards/token-cost.json
```

Expected: compile exit 0; all tests PASS; both JSON files valid.

- [ ] **Step 2: Verify skill tools against the live collector (no LLM needed)**

Run a short Python script from the repository root:

```python
import os, sys
sys.path.insert(0, "app")
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"
from context import set_run_context
import skills
import telemetry

set_run_context("verify-user", "verify-conv")
skills.load_skill("travel_policy")
skills.search_skill_resources("travel_policy", "Japan visa", top_k=2)
skills.run_skill_script("travel_policy", "scripts/estimate_budget.py",
                        ["--city", "Tokyo", "--days", "3", "--style", "budget"])
telemetry.provider.force_flush()
telemetry.meter_provider.force_flush()
```

Then query Tempo for the traces and their span names:

```bash
END=$(date +%s); START=$((END-600))
for id in $(curl -sG 'http://127.0.0.1:3200/api/search' \
  --data-urlencode 'q={ span.gen_ai.operation.name = "execute_tool" }' \
  --data-urlencode "start=$START" --data-urlencode "end=$END" \
  | jq -r '.traces[].traceID'); do
  curl -s -H 'Accept: application/json' "http://127.0.0.1:3200/api/traces/$id" \
    | jq -r '.. | objects | select(.name?) | .name'
done | sort -u
```

Expected: `execute_tool load_skill travel_policy`,
`execute_tool search_skill_resources`,
`execute_tool run_skill_script travel_policy scripts/estimate_budget.py`, and
`retrieval travel-policy-docs` (the retrieval span is a child of the search
tool span).

- [ ] **Step 3: Start the LLM gateway and run the workflow**

The gateway is currently down. Start it (the user knows the launcher; the
observed data lives under `/root/.local/share/copilot-api`). Then run:

```bash
POC_RUNS=4 POC_INTERVAL=2 \
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
python3 app/langgraph_app.py
```

Expected: four completed runs. Conversation 1 runs first, then conversation 2;
each conversation's second run contains `search_memory` with a non-zero record
count, and the first run contains `create_memory`.

- [ ] **Step 4: Verify the workflow root and paths**

Query Tempo for the latest traces and check:

- the root span `invoke_workflow travel_workflow` has `gen_ai.operation.name`,
  `gen_ai.workflow.name`, `gen_ai.conversation.id`, `enduser.id`;
- it has **no** `gen_ai.usage.*` and no `poc.*`;
- the policy run contains `execute_tool load_skill travel_policy`,
  `execute_tool search_skill_resources`, a child `retrieval travel-policy-docs`,
  and `execute_tool run_skill_script travel_policy scripts/estimate_budget.py`;
- the second run in a conversation contains `search_memory` with
  `gen_ai.memory.record.count > 0`;
- failed spans (if any) carry `error.type` and an ERROR status.

- [ ] **Step 5: Verify metrics**

Query Prometheus:

```bash
for m in gen_ai_client_operation_duration_seconds_count \
         gen_ai_client_inference_time_per_output_chunk_seconds_count \
         gen_ai_execute_tool_duration_seconds_count \
         gen_ai_invoke_agent_tool_calls_count; do
  echo "== $m"
  curl -s "http://127.0.0.1:9091/api/v1/query?query=$m" | jq -r '.data.result | length'
done
```

Expected: each count is > 0 after the run.

Check that the span-metric labels no longer include conversation/user:

```bash
grep -n 'gen_ai.conversation.id\|enduser.id' tempo/tempo.yaml
curl -s 'http://127.0.0.1:9091/api/v1/label/__name__/values' | jq -r '.data[]' | grep traces_spanmetrics
curl -sG 'http://127.0.0.1:9091/api/v1/query' \
  --data-urlencode 'query=count by (gen_ai_conversation_id)(traces_spanmetrics_calls_total)' | jq -r '.data.result'
```

Expected: the `grep` returns no matches (the config is authoritative for new
series). The last query may still return old series until Prometheus marks them
stale; do not use it alone to judge the config. The Tempo dimensions no longer
include the two labels.

- [ ] **Step 6: Verify dashboards**

Use the Grafana API to confirm both dashboards load and the new panels exist:

```bash
curl -s http://127.0.0.1:3000/api/dashboards/uid/agent-insight \
  | jq -r '.dashboard.panels[] | select(.title=="Runs" or .title=="Conversations" or .title=="Scenarios" or .title=="Operation mix") | .title'
curl -s http://127.0.0.1:3000/api/dashboards/uid/token-cost \
  | jq -r '.dashboard.panels[] | select(.title=="Total Cost" or .title=="Cost over time") | .title'
```

Expected: all titles present. Inspect the rendered dashboard in the browser
(`http://localhost:3000`) and check the Scenarios row, the Conversations table,
and the cost KPIs. Do not mutate the provisioned dashboard for testing.

- [ ] **Step 7: Report**

Report:
- files changed;
- test/validation output;
- the exact span names seen in Tempo;
- the metric series counts;
- any check blocked by the gateway or environment.

---

## Self-review

- [ ] **Spec coverage:** Agent Skills (load / read / command), retrieval, memory,
  content switch, no `poc.*`, no conversation/user metric dimensions, no cost
  metric, dashboard terminology, cost derived inline, Tempo dimensions, README.
- [ ] **Placeholder scan:** no TBD/TODO; every code step has code; every command
  has expected output.
- [ ] **Type consistency:** `execute_tool_span` yields an OTel span; `skills`
  functions use it; `memory` uses `tracer` and `record_operation_duration`;
  `langgraph_app` imports the exact names defined in `telemetry`/`skills`/`memory`.
- [ ] **Standard precedence:** the implementation never writes
  `gen_ai.usage.*` on `invoke_workflow`, never adds conversation/user metric
  dimensions, and never emits `poc.*`.
