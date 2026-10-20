# AI Observability PoC - LangGraph x OpenTelemetry GenAI

A LangGraph travel assistant instrumented strictly per the OpenTelemetry **GenAI
semantic conventions** and the internal AI observability standard
(`ai-observability-design.md` / `ai-observability-standard.zh.md`). The app emits
standard spans and metrics over OTLP; Grafana dashboards are provisioned from
`grafana/dashboards/`.

```
LangGraph agent --OTLP--> OTel Collector --> Tempo (traces) + Prometheus (metrics) --> Grafana
```

### Telemetry path

| Hop | Endpoint | Data |
|---|---|---|
| LangGraph app | OTLP gRPC `localhost:4317` | spans and metrics |
| OTel Collector | receiver `:4317`; trace exporter `tempo:4317`; metrics exporter `:8889`; self-metrics `:8888` | batches and routes |
| Tempo | OTLP ingest `tempo:4317` (internal); HTTP API `:3200` | traces; span metrics remote-written to Prometheus |
| Prometheus | host `:9091` | scrapes `otel-collector:8889` and receives Tempo span metrics |
| Grafana | `:3000` | Tempo + Prometheus datasources |

The app never talks to Tempo or Prometheus directly. Tempo's OTLP port is not
published to the host, so every span goes through the collector. The collector's
Prometheus exporter is the only ingress for app-emitted metrics.

Everything in this repository is **English only**, except the official Chinese
standard document. The official standard files are not modified by the example.

## Code layout

The demo uses explicit dependencies instead of global telemetry state:

| Module | Responsibility |
|---|---|
| `app/config.py` | `Settings` (environment, models, paths) |
| `app/context.py` | `RunContext` carried by a `ContextVar` per run |
| `app/models.py` | `TokenUsage`, `InferenceResult`, `Skill`, `SearchHit`, `MemoryRecord` |
| `app/telemetry.py` | `Telemetry` facade: spans, metrics, content switch, serialization |
| `app/llm.py` | `ChatClient`: streaming inference plus inference telemetry |
| `app/tools.py` | `HttpClient`, `WebTools` (weather / currency) |
| `app/skills.py` | `SkillRepository` (files/search/scripts) and `SkillTools` |
| `app/memory.py` | `JsonMemoryStore` and `MemoryService` |
| `app/prompts.py` | Prompt constants |
| `app/graph.py` | `AgentSpec`, supervisor/specialist/writer nodes, graph builder |
| `app/workflow.py` | Conversations, `WorkflowRunner`, dependency wiring, `main` |
| `app/langgraph_app.py` | Entry point |

`Telemetry` owns its OTel providers and is injected into the LLM client, tools,
memory service, and graph. Tests construct it with in-memory exporters.

---

## Dashboards

| UID | Title | File | Purpose |
|-----|-------|------|---------|
| `agent-insight` | Agent Insight | `agent-insight.json` | Errors (by operation, type, and bounded source), the trace-derived Conversations card, Run volume, the Trajectory table, the Percentiles row (p95/p99), and the Scenarios row (skills, retrieval, memory). |
| `token-cost` | Token & Cost | `token-cost.json` | Cost KPIs, spend trends, and cost by model / agent. Cost is derived in the dashboard from tokens × prices; there is no cost metric. |

Both dashboards share the `application` → `agent` → `model` cascade. Agent
Insight also has a `skill` filter.

### Conversations and users are trace-only

`gen_ai.conversation.id` and `enduser.id` are unbounded identifiers. Per standard
R-11.1-01 / R-8.3-02 they **never enter metric dimensions**, including Tempo
span-metrics. The Conversations KPI card reads workflow spans with TraceQL and
counts distinct conversations without a metric dimension.

The **Activity over time** panel uses bounded run/chat/tool counts, and
**Run volume** shows workflow runs by success/error status. `gen_ai.conversation.id`
and `enduser.id` are not Tempo span-metric dimensions; the Trajectory table and
the Conversations KPI card remain trace-only. Tempo 2.4.1 has no TraceQL metrics
endpoint, so a distinct-conversation time series is intentionally not offered;
use the card and the trace table for conversation drill-down.

### Errors

The **Errors** row breaks errors down by operation, error type, and top bounded
source labels (operation / agent / model / tool / error type). It never promotes
conversation or user ids into a metric dimension.

The **Request Topology** panel is the built-in nodeGraph (Supervisor agent →
Agents → Models). The full entity graph exceeds Grafana's hidden-node threshold;
tool calls live in the Tools row. Inspect it in a real browser — headless
screenshots can show “Computing layout” before the layout worker finishes.

Telemetry attribute names remain the standard `gen_ai.*` names; the dashboards
use human-readable display labels (`Model:`, `Agent:`, `Tool:`, `Operation:`)
and never expose the raw attribute key as a visible column or legend.

### Percentiles

The Percentiles row charts p95 and p99 for inference, agent, tool, TTFT, and
retrieval/memory operations. Inference, agent, and TTFT use the app-emitted
histograms; tool and scenario percentiles use Tempo span-metric latency so short
demo runs still render. Use the model/agent/skill filters to isolate the relevant
lines.

The **Models** row also charts **TPOT** (time per output token), derived
query-side as `(inference duration − TTFT) / output tokens` in ms/token. TPOT is
a derived signal, not a new metric; it complements TTFT by separating generation
speed from start latency.

### Cost is query-side

`token-cost` derives cost from the cumulative token counters and an inline price
table embedded in its PromQL (USD per 1M input/output tokens):

| Model | Input | Output |
|---|---|---|
| `gemini-3.8-flash` | 0.3 | 2.5 |
| `gpt-4o` | 2.5 | 10.0 |
| `kimi-k3` | 3.0 | 15.0 |

This follows the standard's derived-signal definition: cost = token usage × unit
price, computed at query time. It is not a telemetry metric and no local extension
metric is created. Update the inline price vectors when models or prices change;
unknown models contribute zero cost.

---

## Scenarios

```
invoke_workflow travel_workflow
├─ search_memory                          INTERNAL
├─ invoke_agent supervisor_agent          INTERNAL
│  └─ chat {model}                        CLIENT
├─ invoke_agent weather_agent             INTERNAL
│  ├─ chat {model}                        CLIENT
│  └─ execute_tool get_weather            INTERNAL
├─ invoke_agent currency_agent            INTERNAL
│  ├─ chat {model}                        CLIENT
│  └─ execute_tool convert_currency       INTERNAL
├─ invoke_agent policy_agent              INTERNAL
│  ├─ chat {model}                        CLIENT
│  ├─ execute_tool load_skill travel_policy
│  ├─ execute_tool search_skill_resources INTERNAL
│  │  └─ retrieval travel-policy-docs     CLIENT
│  ├─ execute_tool read_skill_resource travel_policy references/...
│  ├─ execute_tool estimate_budget travel_policy scripts/estimate_budget.py
│  └─ chat {model}                        CLIENT
├─ invoke_agent writer_agent              INTERNAL
│  └─ chat {model}                        CLIENT
└─ create_memory                          INTERNAL
```

- **Agent Skill**: `app/skills/travel_policy/` contains `SKILL.md`,
  `references/` and `scripts/estimate_budget.py`. The policy agent loads the
  skill, reads references, and calls the dedicated `estimate_budget` tool, which
  executes the skill script as a real subprocess. `run_skill_script` remains the
  generic script tool for other scripts. All file reads and script executions are
  real.
- **Retrieval**: a real local TF-IDF search over the skill references. The
  retrieval span carries `gen_ai.data_source.id=travel-policy-docs` and
  `gen_ai.retrieval.top_k`; query text and hit documents are Opt-In content.
- **Memory**: a local JSON store (`POC_MEMORY_PATH`, default
  `app/.state/memory.json`) keyed by `gen_ai.conversation.id`. The first run of a
  conversation writes a Q/A record; the second run reads it back. Writes are
  atomic and malformed files do not fail the run.
- **Content switch**: messages, tool arguments/results, retrieval query/documents,
  memory query/records and tool definitions are emitted only when
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`. The default is off.
  Structural spans and metrics are unaffected by the switch.

## Spans

| Operation | Span name | Kind |
|---|---|---|
| `invoke_workflow` | `invoke_workflow travel_workflow` | INTERNAL |
| `invoke_agent` | `invoke_agent {agent}` | INTERNAL |
| `chat` | `chat {model}` | CLIENT |
| `execute_tool` | `execute_tool {tool}` | INTERNAL |
| `execute_tool` (load skill) | `execute_tool load_skill travel_policy` | INTERNAL |
| `execute_tool` (read resource) | `execute_tool read_skill_resource travel_policy {resource}` | INTERNAL |
| `execute_tool` (command) | `execute_tool estimate_budget travel_policy scripts/estimate_budget.py` | INTERNAL |
| `retrieval` | `retrieval travel-policy-docs` | CLIENT |
| `search_memory` | `search_memory` | INTERNAL |
| `create_memory` | `create_memory` | INTERNAL |

Span names never contain questions, ids, or other dynamic content.

## Metrics

Client-side standard instruments emitted by the app:

| Semantic convention name | Prometheus name (base) | Type | Unit |
|---|---|---|---|
| `gen_ai.client.inference.duration` | `gen_ai_client_inference_duration_seconds` | Histogram | s |
| `gen_ai.client.inference.time_to_first_chunk` | `gen_ai_client_inference_time_to_first_chunk_seconds` | Histogram | s |
| `gen_ai.client.inference.time_per_output_chunk` | `gen_ai_client_inference_time_per_output_chunk_seconds` | Histogram | s |
| `gen_ai.client.inference.usage.input_tokens` | `gen_ai_client_inference_usage_input_tokens_total` | Counter | {token} |
| `gen_ai.client.inference.usage.output_tokens` | `gen_ai_client_inference_usage_output_tokens_total` | Counter | {token} |
| `gen_ai.client.inference.usage.cache_read.input_tokens` | `gen_ai_client_inference_usage_cache_read_input_tokens_total` | Counter | {token} |
| `gen_ai.client.inference.usage.cache_write.input_tokens` | `gen_ai_client_inference_usage_cache_write_input_tokens_total` | Counter | {token} |
| `gen_ai.client.inference.usage.reasoning.output_tokens` | `gen_ai_client_inference_usage_reasoning_output_tokens_total` | Counter | {token} |
| `gen_ai.client.inference.operation.input_tokens` | `gen_ai_client_inference_operation_input_tokens` | Histogram | {token} |
| `gen_ai.client.inference.operation.output_tokens` | `gen_ai_client_inference_operation_output_tokens` | Histogram | {token} |
| `gen_ai.client.operation.duration` | `gen_ai_client_operation_duration_seconds` | Histogram | s |
| `gen_ai.invoke_workflow.duration` | `gen_ai_invoke_workflow_duration_seconds` | Histogram | s |
| `gen_ai.invoke_agent.duration` | `gen_ai_invoke_agent_duration_seconds` | Histogram | s |
| `gen_ai.invoke_agent.inference_calls` | `gen_ai_invoke_agent_inference_calls` | Histogram | {inference_call} |
| `gen_ai.invoke_agent.tool_calls` | `gen_ai_invoke_agent_tool_calls` | Histogram | {tool_call} |
| `gen_ai.execute_tool.duration` | `gen_ai_execute_tool_duration_seconds` | Histogram | s |

`gen_ai.client.operation.duration` covers retrieval and memory operations; it is
not used for inference. On failure, duration metrics carry `error.type` and the
per-operation token histograms are recorded with value 0.

There are **no local extension metrics** in this reference implementation.

## Attributes

| Attribute | Applied to |
|---|---|
| `gen_ai.operation.name` | Every GenAI span |
| `gen_ai.provider.name` | `chat` |
| `gen_ai.conversation.id` | All spans when available |
| `enduser.id` | All spans when available; Required on `invoke_workflow` |
| `gen_ai.workflow.name` | `invoke_workflow` |
| `gen_ai.agent.name` / `gen_ai.agent.description` | `invoke_agent`, `execute_tool` |
| `gen_ai.request.model` / `gen_ai.response.model` | `chat`; request model also on single-model `invoke_agent` and on the workflow root for the legacy Trajectory filter |
| `gen_ai.response.id` / `gen_ai.response.finish_reasons` / `gen_ai.response.time_to_first_chunk` | `chat` |
| `gen_ai.usage.*` | `chat`, `invoke_agent` |
| `gen_ai.tool.name` / `type` / `call.id` / `description` | `execute_tool` |
| `gen_ai.skill.name` / `description` / `source.uri` / `resource.name` | Skill refinements |
| `process.executable.name` / `path`, `process.exit.code` | Command execution refinement |
| `gen_ai.data_source.id`, `gen_ai.retrieval.top_k` | `retrieval` |
| `gen_ai.retrieval.query.text` / `documents` | `retrieval` (Opt-In) |
| `gen_ai.memory.store.id` / `record.id` / `record.count` | Memory spans |
| `gen_ai.memory.query.text` / `records` | Memory spans (Opt-In) |
| `error.type` | Any failed operation |
| `gen_ai.input.messages` / `output.messages` / `system_instructions` / `tool.definitions` | Opt-In content |
| `gen_ai.tool.call.arguments` / `result` | `execute_tool` (Opt-In) |

The restored legacy Trajectory table uses per-run `gen_ai.usage.*` totals and the
average `gen_ai.response.time_to_first_chunk` on `invoke_workflow`, plus the legacy
`poc.ttft` / `poc.error_count` / `poc.inference_calls` / `poc.tool_calls` fields.
These are **demo-only** additions for the old single-table view; the standard
contract keeps usage on `chat` / `invoke_agent` and does not define `poc.*`.

## Running

Prerequisites: Docker (Collector + Tempo + Prometheus + Grafana) and an
OpenAI-compatible LLM gateway.

```bash
# 1. Start the observability stack
docker compose up -d

# 2. Configure the app
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export POC_LLM_BASE_URL=http://127.0.0.1:4141/v1
export POC_MODELS=gemini-3.8-flash,gpt-4o,kimi-k3
export POC_RUNS=6
export POC_INTERVAL=3
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

# 3. Run the agent (terminal 1)
python3 app/langgraph_app.py

```

`POC_RUNS=0` runs until interrupted. The first run of a conversation has no
memory; the second run reads the first run's Q/A from the memory store.

Ports: Grafana `:3000`, Prometheus `:9091`, Tempo `:3200`, OTLP `:4317` (gRPC) /
`:4318` (HTTP), Collector self-metrics `:8888`.

### Tempo dimensions

The span-metrics generator promotes bounded dimensions only:

`gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.provider.name`,
`gen_ai.tool.name`, `gen_ai.agent.name`, `gen_ai.workflow.name`,
`gen_ai.skill.name`, `gen_ai.skill.resource.name`, `gen_ai.data_source.id`,
`gen_ai.memory.store.id`, `error.type`.

Changing the list requires a **Tempo restart** (`docker compose restart tempo`).
`gen_ai.conversation.id`, `enduser.id` and `gen_ai.tool.call.id` are deliberately
absent.

### Notes

- The app forces CUMULATIVE temporality for counters and histograms; DELTA series
  break `rate()`/`increase()` after the OTLP → Prometheus mapping.
- Cache telemetry depends on the upstream model. `kimi-k3` and `gpt-4o` report
  cache reads (a repeated prompt prefix is served from cache);
  `gemini-3.8-flash` on the local Copilot gateway returns `cached_tokens=0` even
  for an identical prefix. A constant 0 is missing data, not a true zero hit rate.
- Every histogram has explicit bucket boundaries.
- If the Docker compose bridge loses its gateway address after a host reboot,
  restore it with `ip addr add 172.18.0.1/16 dev <bridge>`.
