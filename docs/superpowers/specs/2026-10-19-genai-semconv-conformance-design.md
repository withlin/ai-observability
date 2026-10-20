# GenAI Semantic-Conventions Conformance Refactor

**Status**: Approved
**Date**: 2026-10-19
**Standard**: `ai-observability-design.md` v1 (OpenTelemetry GenAI semantic conventions)
**Scope**: `app/langgraph_app.py`, `grafana/dashboards/agent-insight.json`,
`grafana/dashboards/token-cost.json`, `tempo/tempo.yaml`,
`prometheus/prometheus.yml`, `prometheus/rules/genai-cost.yml`, `README.md`

## 1. Context

The current demo emits the basic workflow / agent / chat / tool signal set, but it
also emits local `poc.*` extensions that the standard bans as non-conformant
examples. It covers none of the standard's extended scenarios. The two
provisioned dashboards depend on the local cost counter and use terminology that
does not match the standard's glossary.

This refactor makes the demo a clean reference implementation:

- adds Agent Skills (all three `execute_tool` refinements), retrieval, and memory;
- removes every `poc.*` signal and attribute;
- aligns span names, kinds, attributes, metrics, and dashboard language with the
  standard;
- derives cost query-side instead of emitting a cost metric.

## 2. Goals and non-goals

### Goals

1. Every emitted GenAI span name, operation, attribute, and metric name exists in
   the standard. No invented `gen_ai.*` names.
2. Add a real Agent Skill workflow: load skill, read skill resource, run a skill
   script, and retrieve from the skill's document collection.
3. Add real cross-run memory keyed by `gen_ai.conversation.id`.
4. Remove all `poc.*` metrics, span attributes, and dashboard references.
5. Keep both dashboards useful after the cleanup; cost is derived from tokens.
6. Keep every scenario real: real files, real script execution, real retrieval,
   real LLM calls. No simulated latency, errors, or random data.

### Non-goals

- `embeddings`: the local gateway does not support embeddings; the demo does not
  fake an embedding model.
- `plan`, `create_agent`, `fetch_response`, server-side metrics: not part of the
  selected scope or not applicable to this client application.
- The three GenAI events (log signal): the repo has no log backend. Content stays
  on span attributes, which the standard explicitly allows as one of three
  content placements.
- Alert thresholds, SLOs, and outcome-quality evaluation.

## 3. Scenario and graph

The supervisor graph gains one specialist:

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
│  ├─ execute_tool run_skill_script travel_policy scripts/estimate_budget.py
│  └─ chat {model}                        CLIENT
├─ invoke_agent writer_agent              INTERNAL
│  └─ chat {model}                        CLIENT
└─ create_memory                          INTERNAL
```

- The supervisor routes to `weather_agent`, `currency_agent`, `policy_agent`, or
  `writer`. The existing specialists and writer remain.
- `policy_agent` is a ReAct subgraph (`bind_tools` + `ToolNode`) whose tools are
  the Agent Skill tools plus the retrieval tool. The model decides which tools to
  call; nothing is hardcoded.
- Memory is read before the graph run and written after the writer finishes, so
  the second run in a conversation can use the first run's result.
- The trace shows only operations that actually occurred; a run that asks only
  about weather will not contain policy, retrieval, or script spans.

## 4. Application architecture

### 4.1 Modules

`app/langgraph_app.py` stays the entry point (`python3 app/langgraph_app.py`)
and owns the graph, prompts, LLM calls, and the run loop. The pieces that would
make it too large move to focused modules:

| File | Responsibility |
|---|---|
| `app/telemetry.py` | Resource, tracer/meter providers, instruments, histogram views, content-capture switch, message/JSON serialization helpers |
| `app/skills.py` | Agent Skill package loading, resource reads with path confinement, local TF-IDF retrieval, script execution with timeout, instrumented skill/retrieval tools |
| `app/memory.py` | Local JSON memory store, atomic writes, `search_memory` / `create_memory` spans and operation metrics |
| `app/langgraph_app.py` | Graph, supervisor/policy/writer nodes, LLM instrumentation, conversation scripts, main loop |

Imports are sibling imports (`from telemetry import ...`), which works when the
script is run directly because Python adds `app/` to `sys.path`.

### 4.2 Agent Skill package

```
app/skills/travel_policy/
  SKILL.md
  references/entry_requirements.md
  references/baggage_rules.md
  references/budget_guide.md
  scripts/estimate_budget.py
```

- `SKILL.md` has YAML frontmatter with `name: travel_policy` and a description,
  followed by instructions. The loader parses the frontmatter without an
  external service.
- `load_skill(skill_name)` reads `SKILL.md` and returns it to the model.
- `search_skill_resources(skill_name, query, top_k=5)` performs a real local
  TF-IDF cosine-similarity search over `references/*.md`. Document ids are
  skill-relative paths; scores are cosine similarities. No vector database and
  no embedding model are implied.
- `read_skill_resource(skill_name, resource_name)` reads a skill-relative file
  after resolving it and verifying it stays inside the skill directory.
- `run_skill_script(skill_name, script_name, args)` executes
  `scripts/estimate_budget.py` with `sys.executable`, `shell=False`, a 20-second
  timeout, and the skill directory as `cwd`. The script returns JSON with daily
  and total budget ranges for a city, number of days, and style.
- The skill source URI is `file://<absolute path to the skill directory>`.
- The tools are ordinary Python functions wrapped by the existing
  `instrumented_tool` helper, extended to support the skill refinement
  attributes and span names.

### 4.3 Conversation and memory lifecycle

The run loop uses three two-question conversation scripts:

| Conversation | User | First run | Second run |
|---|---|---|---|
| `conv-0001` | `user-001` | Tokyo weather + EUR→JPY | Same Tokyo trip: visa, baggage, 5-day mid-range budget |
| `conv-0002` | `user-002` | Lisbon weather + USD→EUR | Same Lisbon trip: visa, baggage, 4-day budget |
| `conv-0003` | `user-003` | Toronto weather + USD→CAD | Same Toronto trip: visa, baggage, 3-day budget |

Run `i` maps to conversation `((i-1) // 2) % 3` and question `(i-1) % 2`.
`POC_RUNS=0` loops forever. The model still cycles per run.

Memory store:

- Default path `app/.state/memory.json`, overridable with `POC_MEMORY_PATH`.
- Top-level object `{"traveler_memory": [record, ...]}`.
- A record contains `id`, `content` (question and answer), and `metadata`
  (`conversation_id`, `user_id`, `created_at`).
- Writes are atomic (write a temporary file, then `os.replace`).
- `search_memory` filters by `conversation_id`, returns the most recent records,
  and does not fail the run if the file is missing or malformed; the span
  records the error and the run continues with empty memory.
- When records exist, they are injected as a `SystemMessage` before the user
  question, so the supervisor, specialists, and writer can use them.
- The span attributes record structure (`store.id`, `record.count`, `record.id`)
  regardless of the content switch; query text and record bodies are Opt-In.

### 4.4 Content capture switch

- One switch: `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` (accepted
  truthy values: `1`, `true`, `yes`, `on`). Default is off.
- When off, the following attributes are omitted; no structural span or metric
  changes:
  - `gen_ai.input.messages`, `gen_ai.output.messages`,
    `gen_ai.system_instructions`;
  - `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`;
  - `gen_ai.retrieval.query.text`, `gen_ai.retrieval.documents`;
  - `gen_ai.memory.query.text`, `gen_ai.memory.records`.
- The README run command enables the switch explicitly for the local demo.
- On a failed operation, do not synthesize `gen_ai.output.messages`.

## 5. Telemetry contract

### 5.1 Spans

| Operation | Span name | Kind | Notes |
|---|---|---|---|
| `invoke_workflow` | `invoke_workflow travel_workflow` | INTERNAL | Root of each run |
| `invoke_agent` | `invoke_agent {agent}` | INTERNAL | In-process agents |
| `chat` | `chat {model}` | CLIENT | One streamed model call |
| `execute_tool` | `execute_tool {tool}` | INTERNAL | Base tool form |
| `execute_tool` (load skill) | `execute_tool load_skill travel_policy` | INTERNAL | Skill name available |
| `execute_tool` (read resource) | `execute_tool read_skill_resource travel_policy references/entry_requirements.md` | INTERNAL | Skill + resource names available |
| `execute_tool` (command) | `execute_tool run_skill_script travel_policy scripts/estimate_budget.py` | INTERNAL | Skill + resource names available |
| `retrieval` | `retrieval travel-policy-docs` | CLIENT | Child of `search_skill_resources` |
| `search_memory` | `search_memory` | INTERNAL | In-process memory system |
| `create_memory` | `create_memory` | INTERNAL | In-process memory system |

Span names must not contain dynamic content (questions, ids). The skill and tool
names are bounded identifiers.

### 5.2 New span attributes

All GenAI spans carry `gen_ai.operation.name` and, when available,
`gen_ai.conversation.id` and `enduser.id`.

Skill spans:

| Attribute | Level | Applies to |
|---|---|---|
| `gen_ai.tool.name` | Required | All execute_tool spans |
| `gen_ai.tool.type` | Recommended | All execute_tool spans; value `function` |
| `gen_ai.tool.description` | Recommended | All execute_tool spans |
| `gen_ai.tool.call.id` | Recommended | All execute_tool spans |
| `gen_ai.tool.call.arguments` / `result` | Opt-In | All execute_tool spans |
| `gen_ai.agent.name` | Conditionally Required | All execute_tool spans |
| `gen_ai.skill.name` | Conditionally Required | Load / read / command refinements |
| `gen_ai.skill.description` | Recommended | Load / read / command refinements |
| `gen_ai.skill.source.uri` | Recommended | Load / read / command refinements |
| `gen_ai.skill.resource.name` | Conditionally Required | Read and command refinements |
| `process.executable.name` | Conditionally Required | Command refinement |
| `process.executable.path` | Recommended | Command refinement |
| `process.exit.code` | Conditionally Required | Command refinement |
| `error.type` | Conditionally Required | On failure |

Retrieval span:

| Attribute | Level |
|---|---|
| `gen_ai.operation.name=retrieval` | Required |
| `gen_ai.data_source.id=travel-policy-docs` | Conditionally Required |
| `gen_ai.retrieval.top_k` | Recommended |
| `gen_ai.retrieval.query.text` | Opt-In |
| `gen_ai.retrieval.documents` | Opt-In (JSON string of `{id, score}` list) |
| `gen_ai.conversation.id`, `enduser.id` | Conditionally Required |
| `error.type` | Conditionally Required |

Local retrieval has no external provider or server peer, so
`gen_ai.provider.name`, `server.address`, and `server.port` are omitted.

Memory spans:

| Attribute | Level |
|---|---|
| `gen_ai.operation.name=search_memory or create_memory` | Required |
| `gen_ai.memory.store.id=traveler_memory` | Conditionally Required |
| `gen_ai.memory.record.id` | Conditionally Required for `create_memory` |
| `gen_ai.memory.record.count` | Recommended |
| `gen_ai.memory.query.text` | Opt-In (`search_memory`) |
| `gen_ai.memory.records` | Opt-In (JSON string) |
| `gen_ai.conversation.id`, `enduser.id` | Conditionally Required |
| `error.type` | Conditionally Required |

In-process memory uses Kind `INTERNAL`, which the standard permits.

### 5.3 Existing span corrections

- `invoke_workflow` carries `gen_ai.workflow.name`, `gen_ai.conversation.id`,
  `enduser.id`, per-run `gen_ai.usage.*` totals (contract-level aggregate),
  Opt-In input/output messages, and `error.type` on failure. It does not carry
  `gen_ai.request.model` or any `poc.*` attribute.
- `invoke_agent` carries `gen_ai.agent.name`, `gen_ai.agent.description`,
  `gen_ai.request.model`, `gen_ai.conversation.id`, `enduser.id`,
  `gen_ai.tool.definitions` when tools are bound, Opt-In
  `gen_ai.system_instructions` / messages, usage totals, and `error.type` on
  failure.
- `chat` keeps `gen_ai.provider.name`, request/response model, temperature,
  stream flag, `server.address`/`server.port`, conversation id, enduser id,
  finish reasons, usage, and error attributes. It no longer carries
  `gen_ai.agent.name`; the parent agent span identifies the agent. The inference
  metrics still carry `gen_ai.agent.name` as required by the standard's metric
  dimension rules.
- On a failed inference, record the duration metric and the per-operation token
  histograms with value 0 and `error.type`; do not increment usage counters and
  do not synthesize output messages.
- The workflow, agent, and tool duration metrics record `error.type` on failure.

### 5.4 Metrics

Keep:

- `gen_ai.client.inference.duration`
- `gen_ai.client.inference.time_to_first_chunk`
- `gen_ai.client.inference.usage.input_tokens`
- `gen_ai.client.inference.usage.output_tokens`
- `gen_ai.client.inference.usage.cache_read.input_tokens`
- `gen_ai.client.inference.usage.cache_write.input_tokens`
- `gen_ai.client.inference.usage.reasoning.output_tokens`
- `gen_ai.client.inference.operation.input_tokens`
- `gen_ai.client.inference.operation.output_tokens`
- `gen_ai.invoke_workflow.duration`
- `gen_ai.invoke_agent.duration`
- `gen_ai.invoke_agent.inference_calls`
- `gen_ai.invoke_agent.tool_calls`
- `gen_ai.execute_tool.duration`

Add:

- `gen_ai.client.inference.time_per_output_chunk` (Histogram, s) — interval
  between streamed chunks after the first.
- `gen_ai.client.operation.duration` (Histogram, s) — retrieval and memory
  operations, dimensioned by `gen_ai.operation.name`, optionally
  `gen_ai.agent.name`, and `error.type` on failure.

Remove:

- `poc.cost.usd`
- `poc.inference.errors`

Every histogram keeps an explicit bucket view. No metric uses
`gen_ai.conversation.id`, `enduser.id`, or `gen_ai.tool.call.id` as a dimension.

## 6. Dashboards

### 6.1 Agent Insight

- KPI cards: `Runs`, `Conversations`, `Users`, `Steps`, `Tokens`,
  `Cache Hit Rate`, `Avg Duration`, `Error Rate`.
  - Runs: `invoke_workflow` span count.
  - Conversations: distinct `gen_ai.conversation.id`.
  - Users: distinct `enduser.id`.
  - Steps: `chat` + `execute_tool` span count, per the standard's step definition.
  - Error Rate: failed workflow spans divided by all workflow spans.
- Trends: activity over time (Runs / Conversations / Users), token usage, latency.
- New `Scenarios` row:
  - skill calls over time by skill and tool;
  - skill operation p95;
  - retrieval calls and p95 by data source;
  - memory operations and p95 by operation;
  - a skill table with loads, resource reads, script runs, and p95.
- New `Operation mix` panel showing every emitted `gen_ai.operation.name`.
- Trajectory table: Trace ID, Input, Output, Start, Duration, Total Tokens,
  Application, User ID, Conversation ID, Status, Spans. Remove LLM Calls,
  Tool Calls, and every `poc.*` column. Input/Output are blank when content
  capture is off.
- Tools and Models rows switch error panels to `error.type`; the Models row adds
  time-per-output-chunk. Remove the Cost row.
- Variables: `application` → `agent` → `model`, plus `skill`.
- Every new or changed panel has a short title and a description that says when
  to look at it and what an anomaly looks like.

### 6.2 Token & Cost

- All `poc_cost_usd_total` queries are replaced by
  `app:gen_ai_cost_usd:cumulative`.
- KPI cards: Total Cost, Cost / Hour, Projected / Month, Avg Cost / Request,
  Total Tokens, Cost / 1K Tokens.
- Trends: cost over time stacked by model, cost rate, cost per request.
- Tables: cost share by model, model breakdown, agent breakdown.
- The dashboard description states that cost is derived query-side and is not a
  standard metric.

### 6.3 Cost derivation

Add `prometheus/rules/genai-cost.yml`:

```yaml
groups:
  - name: genai-cost
    interval: 30s
    rules:
      - record: app:gen_ai_cost_usd:cumulative
        expr: |
          sum by (service_name, gen_ai_agent_name, gen_ai_request_model) (
            gen_ai_client_inference_usage_input_tokens_total
              * on (gen_ai_request_model) group_left()
                (
                  label_replace(vector(1.0), "gen_ai_request_model", "claude-haiku-5.5", "", "")
                  or label_replace(vector(2.0), "gen_ai_request_model", "gpt-4.1-2025-04-14", "", "")
                  or label_replace(vector(0.3), "gen_ai_request_model", "gemini-3.8-flash", "", "")
                )
          ) / 1e6
          +
          sum by (service_name, gen_ai_agent_name, gen_ai_request_model) (
            gen_ai_client_inference_usage_output_tokens_total
              * on (gen_ai_request_model) group_left()
                (
                  label_replace(vector(5.0), "gen_ai_request_model", "claude-haiku-5.5", "", "")
                  or label_replace(vector(8.0), "gen_ai_request_model", "gpt-4.1-2025-04-14", "", "")
                  or label_replace(vector(2.5), "gen_ai_request_model", "gemini-3.8-flash", "", "")
                )
          ) / 1e6
```

- Each price vector is a set of `label_replace(vector(price),
  "gen_ai_request_model", "<model>", "", "")` alternatives joined with `or`.
- Prices are deployment-side configuration, not application telemetry. They must
  cover every model in `POC_MODELS`; an unknown model contributes zero cost.
- The initial table matches the app's former `PRICING` values:
  `claude-haiku-5.5` 1/5, `gpt-4.1-2025-04-14` 2/8, `gemini-3.8-flash` 0.3/2.5
  (USD per 1M input/output tokens).
- `prometheus/prometheus.yml` loads `/etc/prometheus/rules/*.yml`;
  `docker-compose.yml` mounts `./prometheus/rules` read-only.

### 6.4 Tempo span-metric dimensions

Add:

- `gen_ai.skill.name`
- `gen_ai.skill.resource.name`
- `gen_ai.data_source.id`
- `gen_ai.memory.store.id`
- `error.type`

Do not add `gen_ai.tool.call.id`. Keep `gen_ai.conversation.id` and `enduser.id`
for the demo-scale KPI cards; document in the README that these are an explicit
cardinality cost and should move to trace queries at scale.

### 6.5 README

Update the span tree, metrics table, attributes table, dashboard table, running
instructions, and the new memory/skill/retrieval sections. Remove the local
extensions table and every `poc.*` reference.

## 7. Verification

Static:

- `python3 -m py_compile app/*.py`
- `jq empty grafana/dashboards/agent-insight.json` and `token-cost.json`
- Parse `prometheus/rules/genai-cost.yml` with `python3 -c 'import yaml; ...'`
  when PyYAML is available
- Search the repo for `poc.` and confirm only historical/gitignored artifacts
  remain (target: zero references in app, dashboards, README).

Instrumentation:

- Call the skill tools directly (`load_skill`, `search_skill_resources`,
  `read_skill_resource`, `run_skill_script`) with the collector running, then
  query Tempo for the exact span names and attributes.
- Start the local LLM gateway and run at least two conversations (four runs).
  Confirm:
  - each run has `invoke_workflow` → `invoke_agent` → `chat` / tool spans;
  - the policy run contains the three skill refinements and a `retrieval` span;
  - the second run contains `search_memory` with a non-zero record count and the
    first run contains `create_memory`;
  - Prometheus exposes the new metrics and the cost recording rule;
  - the dashboards load and the new panels return data.

UI:

- Verify dashboards through the Grafana API and, where possible, a rendered
  screenshot. Do not mutate the provisioned dashboard for testing; use a copy or
  read-only API inspection.
- If the gateway cannot be started, report the blocked end-to-end checks
  explicitly; static, tool-level, and dashboard-JSON checks still run.

## 8. Risks

| Risk | Mitigation |
|---|---|
| The model does not reliably call skill tools | Clear tool descriptions and policy prompt; test with the real gateway; keep the tool set small |
| Tempo/Prometheus restart needed for dimensions and rules | Document the restart; verify after; keep config changes additive |
| Cost price table drifts from `POC_MODELS` | Comment in the rule file and README; unknown models cost zero and are visibly missing |
| Memory file corruption or concurrent runs | Atomic writes, malformed-file recovery, single-threaded run loop |
| Content capture off makes Trajectory Input/Output blank | Document the switch; enable it explicitly in the demo run command |
| `gen_ai.conversation.id` / `enduser.id` span-metric cardinality | Document as a demo-scale tradeoff; never add call ids |

## 9. File inventory

| File | Change |
|---|---|
| `app/langgraph_app.py` | Rewrite graph/run loop; remove `poc.*`; add policy agent and memory lifecycle |
| `app/telemetry.py` | New: OTel setup, instruments, views, content switch, serialization |
| `app/skills.py` | New: skill loader, retrieval, script execution, instrumented tools |
| `app/memory.py` | New: JSON memory store and memory spans |
| `app/skills/travel_policy/SKILL.md` | New |
| `app/skills/travel_policy/references/*.md` | New (3 files) |
| `app/skills/travel_policy/scripts/estimate_budget.py` | New |
| `grafana/dashboards/agent-insight.json` | Retitle KPIs, add Scenarios/Operation mix, remove cost and `poc.*` |
| `grafana/dashboards/token-cost.json` | Switch to derived cost metric |
| `prometheus/rules/genai-cost.yml` | New |
| `prometheus/prometheus.yml` | Load rules |
| `docker-compose.yml` | Mount rules directory |
| `tempo/tempo.yaml` | Add span-metric dimensions |
| `README.md` | Rewrite signal, running, and dashboard documentation |
