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
| LangGraph app | OTLP/HTTP `localhost:4318` (HTTPS for remote collectors) | spans and metrics |
| OTel Collector | receiver `:4318` (HTTP; `:4317` gRPC also enabled); trace exporter `tempo:4317`; metrics exporter `:8889`; self-metrics `:8888` | batches and routes |
| Tempo | OTLP ingest `tempo:4317` (internal); HTTP API `:3200` | traces (TraceQL search and read) |
| Prometheus | host `:9091` | scrapes `otel-collector:8889` (app-emitted metrics) |
| Grafana | `:3000` | Tempo + Prometheus datasources |

The app never talks to Tempo or Prometheus directly. Tempo's OTLP port is not
published to the host, so every span goes through the collector. The collector's
Prometheus exporter is the only ingress for app-emitted metrics.

Everything in this repository is **English only**, except the official Chinese
standard document. The official standard files are not modified by the example.

## Code layout

The demo is a standalone project in `app/langgraph-demo/` with its own
`requirements.txt`. It uses explicit dependencies instead of global telemetry
state:

| Module | Responsibility |
|---|---|
| `app/langgraph-demo/config.py` | `Settings` (environment, models, paths) |
| `app/langgraph-demo/context.py` | `RunContext` carried by a `ContextVar` per run |
| `app/langgraph-demo/models.py` | `TokenUsage`, `InferenceResult`, `Skill`, `SearchHit`, `MemoryRecord` |
| `app/langgraph-demo/telemetry.py` | `Telemetry` facade: spans, metrics, content switch, serialization |
| `app/langgraph-demo/llm.py` | `ChatClient`: streaming inference plus inference telemetry |
| `app/langgraph-demo/tools.py` | `WebTools` (simulated weather / currency data) |
| `app/langgraph-demo/skills.py` | `SkillRepository` (files/search/scripts) and `SkillTools` |
| `app/langgraph-demo/memory.py` | `JsonMemoryStore` and `MemoryService` |
| `app/langgraph-demo/prompts.py` | Prompt constants |
| `app/langgraph-demo/graph.py` | `AgentSpec`, supervisor/specialist/writer nodes, graph builder |
| `app/langgraph-demo/workflow.py` | Conversations, `WorkflowRunner`, dependency wiring, `main` |
| `app/langgraph-demo/langgraph_app.py` | Entry point |

`Telemetry` owns its OTel providers and is injected into the LLM client, tools,
memory service, and graph. Tests construct it with in-memory exporters.

---

## Dashboards

| UID | Title | File | Purpose |
|-----|-------|------|---------|
| `agent-insight` | Agent Insight | `agent-insight.json` | Errors (by operation, type, and bounded source), the trace-derived Conversations card, Run volume, the Trajectory table, the Percentiles row (p95/p99), and the Scenarios row (skill tools, retrieval, memory). |
| `token-cost` | Token & Cost | `token-cost.json` | Cost KPIs, spend trends, and cost by model / agent. Cost is derived in the dashboard from tokens × prices; there is no cost metric. |
| `ai-adoption` | AI Adoption | `ai-adoption.json` | Executive adoption view: applications, agents, models, consumption, growth, health, and a per-application leaderboard with drilldowns. |

All three dashboards share the `application` → `agent` → `model` cascade.

### Importing into another Grafana

All three dashboards declare datasource inputs. Import a JSON via
**Dashboards → New → Import** and select the data sources:

| Dashboard | Inputs |
|---|---|
| `agent-insight.json` | Prometheus (metrics) and Tempo (tracing) |
| `token-cost.json` | Prometheus (metrics) |
| `ai-adoption.json` | Prometheus (metrics) |

Panels and query variables bind to the selected UIDs. The inputs are hidden
template variables, so they add no controls to the dashboard. When provisioned
by this repository they default to the local `prometheus` and `tempo` UIDs.

### Adoption overview

`ai-adoption.json` is the executive adoption view. Applications are counted from
standard telemetry, not an onboarding registry: the value is the number of
services that emitted GenAI metrics in the window. New applications compares the
window with the preceding window of the same length. Agents and models are
distinct standard metric label values. The environment filter defaults to `prod`;
use `All` or `local` for the local demo. Estimated cost uses the same flat
placeholder rates as `token-cost`. The application leaderboard links to Agent
Insight and Token & Cost for each application.

### Conversations and users are trace-only

`gen_ai.conversation.id` and `enduser.id` are unbounded identifiers. Per standard
R-11.1-01 / R-8.3-02 they **never enter metric dimensions**. The Conversations
KPI card reads workflow spans with
TraceQL and counts distinct conversations without a metric dimension.

The **Activity over time** panel uses bounded run/chat/tool counts, and
**Run volume** shows workflow runs by success/error status. `gen_ai.conversation.id`
and `enduser.id` are not metric dimensions; the Trajectory table and
the Conversations KPI card remain trace-only. Tempo 2.4.1 has no TraceQL metrics
endpoint, so a distinct-conversation time series is intentionally not offered;
use the card and the trace table for conversation drill-down.

### Errors

The **Errors** row breaks errors down by operation, error type, and top bounded
source labels (operation / agent / model / tool / error type). It never promotes
conversation or user ids into a metric dimension.

The **Request Topology** panel is a table of agent → model edges with p95
latency. It replaces Grafana's nodeGraph panel, which crashed on graphs with
missing or empty endpoints; tool calls live in the Tools row.

Telemetry attribute names remain the standard `gen_ai.*` names; the dashboards
use human-readable display labels (`Model:`, `Agent:`, `Tool:`, `Operation:`)
and never expose the raw attribute key as a visible column or legend.

### Percentiles

The Percentiles row charts p95 and p99 for inference, agent, tool, TTFT, and
retrieval/memory operations. Inference, agent, and TTFT use the app-emitted
histograms; tool percentiles use `gen_ai.execute_tool.duration` and scenario
percentiles use `gen_ai.client.operation.duration`, so short demo runs still
render. Use the model/agent filters to isolate the relevant
lines.

The **Models** row also charts **TPOT** (time per output token), derived
query-side as `(inference duration − TTFT) / output tokens` in ms/token. TPOT is
a derived signal, not a new metric; it complements TTFT by separating generation
speed from start latency.

### Cost is query-side

`token-cost` derives cost from the cumulative token counters and flat placeholder
rates embedded in its PromQL (USD per 1M tokens): input `1.0`, output `3.0`.
Every model present in the metrics is costed; no model names are hardcoded.

This follows the standard's derived-signal definition: cost = token usage × unit
price, computed at query time. It is not a telemetry metric and no local extension
metric is created. The flat rates are placeholders, not real prices; replace
them with a per-model price catalog (for example, a metric generated from the
gateway's model list) when accurate cost is required.

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

- **Simulated tools**: weather and currency tools return deterministic local
  data and make no external calls. Spans, metrics and the conversion logic are
  identical to a live implementation.
- **Agent Skill**: `app/langgraph-demo/skills/travel_policy/` contains `SKILL.md`,
  `references/` and `scripts/estimate_budget.py`. The policy agent loads the
  skill, reads references, and calls the dedicated `estimate_budget` tool, which
  executes the skill script as a real subprocess. `run_skill_script` remains the
  generic script tool for other scripts. All file reads and script executions are
  real.
- **Retrieval**: a real local TF-IDF search over the skill references. The
  retrieval span carries `gen_ai.data_source.id=travel-policy-docs` and
  `gen_ai.retrieval.top_k`; query text and hit documents are Opt-In content.
- **Memory**: a local JSON store (`POC_MEMORY_PATH`, default
  `app/langgraph-demo/.state/memory.json`) keyed by `gen_ai.conversation.id`.
  The first run of a conversation writes a Q/A record; the second run reads it
  back. Writes are atomic and malformed files do not fail the run.
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
Dashboards read only the standard instruments above; skill names and data
source ids stay in spans rather than becoming metric labels.

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

Prerequisites: Python 3.10+, Docker (Collector + Tempo + Prometheus + Grafana)
and an OpenAI-compatible LLM gateway.

### Dependencies

```bash
python3 -m venv app/langgraph-demo/.venv
. app/langgraph-demo/.venv/bin/activate
pip install -r app/langgraph-demo/requirements.txt
```

For an offline or air-gapped target, download the wheel set on a connected host
and install from it (run the download on the same OS and Python version as the
target):

```bash
pip download -r app/langgraph-demo/requirements.txt -d app/langgraph-demo/wheels/
pip install --no-index --find-links app/langgraph-demo/wheels/ \
  -r app/langgraph-demo/requirements.txt
```

The app exports over OTLP **HTTP/protobuf** only; the gRPC exporter is not a
dependency.

### Local stack

```bash
# 1. Start the observability stack
docker compose up -d

# 2. Configure the app
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export POC_LLM_BASE_URL=http://127.0.0.1:4141/v1
export POC_MODELS=gemini-3.8-flash,gpt-4o,kimi-k3
export POC_RUNS=6
export POC_INTERVAL=3
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

# 3. Run the agent (terminal 1)
python3 app/langgraph-demo/langgraph_app.py

```

### Third-party collector

`OTEL_EXPORTER_OTLP_ENDPOINT` accepts any OTLP/HTTP URL. An `https://` URL
enables TLS with the system CA store; remote collectors do not need the local
Docker stack.

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.example.com:4318
export OTEL_EXPORTER_OTLP_USERNAME=instance-id
export OTEL_EXPORTER_OTLP_PASSWORD=api-token
export OTEL_EXPORTER_OTLP_TLS_SKIP_VERIFY=true   # self-signed endpoint
export OTEL_SERVICE_NAME=travel-assistant
export OTEL_RESOURCE_ATTRIBUTES=deployment.environment=staging
python3 app/langgraph-demo/langgraph_app.py
```

`OTEL_EXPORTER_OTLP_TLS_SKIP_VERIFY=true` disables certificate verification for
self-signed or internal-CA endpoints. Leave it unset to verify against the
system CA store, or point `OTEL_EXPORTER_OTLP_CERTIFICATE` at a CA bundle.

The base URL is extended with `/v1/traces` and `/v1/metrics`. If the provider
publishes separate signal URLs, set the full endpoints instead:

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://otlp.example.com/otlp/v1/traces
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=https://otlp.example.com/otlp/v1/metrics
```

| Variable | Default | Meaning |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | Base OTLP/HTTP URL; signal paths are appended. Use `https://` for TLS. |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `<base>/v1/traces` | Full trace endpoint, used as-is. |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | `<base>/v1/metrics` | Full metric endpoint, used as-is. |
| `OTEL_EXPORTER_OTLP_HEADERS` | empty | Extra headers as comma-separated `key=value` pairs; values may be percent-encoded. |
| `OTEL_EXPORTER_OTLP_TRACES_HEADERS` / `OTEL_EXPORTER_OTLP_METRICS_HEADERS` | empty | Signal-specific headers, merged over the base headers. |
| `OTEL_EXPORTER_OTLP_USERNAME` / `OTEL_EXPORTER_OTLP_PASSWORD` | empty | Basic auth credentials; both must be set. Ignored when an `Authorization` header is supplied. |
| `OTEL_EXPORTER_OTLP_CERTIFICATE` | system CA | Path to a CA bundle for verifying an HTTPS collector. |
| `OTEL_EXPORTER_OTLP_TLS_SKIP_VERIFY` | `false` | Disable HTTPS certificate verification (self-signed endpoints). |
| `OTEL_SERVICE_NAME` | `langgraph-agent` | `service.name` resource attribute. |
| `OTEL_RESOURCE_ATTRIBUTES` | empty | Extra resource attributes (comma-separated `key=value`); can override `deployment.environment` (default `local`) and `service.version` (default `4.0.0`). |

App-level variables:

| Variable | Default | Meaning |
|---|---|---|
| `POC_LLM_BASE_URL` | `http://127.0.0.1:4141/v1` | OpenAI-compatible gateway used for inference. |
| `POC_MODELS` | `gemini-3.8-flash,gpt-4o,kimi-k3` | Comma-separated model rotation. |
| `POC_RUNS` | `1` | Number of workflow runs; `0` runs until interrupted. |
| `POC_INTERVAL` | `0` | Seconds between runs. |
| `POC_PROVIDER_NAME` | `openai` | Value for `gen_ai.provider.name`. |
| `POC_MEMORY_PATH` | `app/langgraph-demo/.state/memory.json` | Local memory store. |
| `POC_SKILLS_ROOT` | `app/langgraph-demo/skills` | Agent Skills root directory. |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `false` | Opt-in message, tool, retrieval and memory content. |

`POC_RUNS=0` runs until interrupted. The first run of a conversation has no
memory; the second run reads the first run's Q/A from the memory store.

Ports: Grafana `:3000`, Prometheus `:9091`, Tempo `:3200`, OTLP/HTTP `:4318`
(the app's exporter), OTLP/gRPC `:4317` (available but unused by the app), and
Collector self-metrics `:8888`.

### Metric dimensions

The dashboards use only the standard metric dimensions. Skill names and data
source ids are span-only refinements; the Skills row groups skill tool calls by
`gen_ai.tool.name`, and Retrieval calls shows the standard operation total. The
local Tempo instance serves traces only and does not derive span metrics or
service graphs.

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
