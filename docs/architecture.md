**Agent Orchestration Architecture**

This document describes the DuckAgent architecture, the orchestration graph for agents, routing rules, storage/compute options, deployment variants (dev vs production), and a recommended implementation plan.

**Diagram (Mermaid)**

```mermaid
flowchart LR
  subgraph UI
    A[Streamlit UI / CLI / Notebook] -->|HTTP / Local API| API[API Layer]
  end

  API --> Router[Router / Intent Classifier]
  Router --> Orchestrator[Orchestrator (Local / LangGraph / AWS Adapter)]

  subgraph Orchestration
    Orchestrator --> Planner[Planner]
    Planner --> SQLGen[SQLGenerator]
    SQLGen --> Validator[Validator (pre-checks)]
    Validator --> SQLRunner[SQLRunner (DuckDB/Athena)]
    SQLRunner --> Analysis[AnalysisAgent]
    Analysis --> Summarizer[Summarizer]
    Summarizer --> ResultStore[Result Store / Cache]
    ResultStore --> API
  end

  subgraph Storage
    SQLRunner --> DuckDB[DuckDB (local) / Parquet on S3]
    SQLRunner --> Athena[Athena]
    ResultStore --> S3[S3 / DynamoDB for traces]
  end

  subgraph LLM
    LLMAdapters[LLM Adapters: OpenAI, Anthropic, Bedrock, Local] --> Planner
    LLMAdapters --> SQLGen
    LLMAdapters --> Summarizer
  end

  subgraph ProdAdapters
    Orchestrator --> StepFunctions[Step Functions / Strands]
    Analysis --> JobRunner[Fargate / Lambda Job Runner]
  end

  classDef infra fill:#f9f,stroke:#333,stroke-width:1px;
  class Storage,ProdAdapters infra
```

Key components
- API Layer: small HTTP interface for Streamlit/CLI/Notebook to submit NL prompts and receive results or job IDs.
- Router: rule-based + optional LLM intent classifier to choose which agents to run and whether to sample or run full analysis.
- Orchestrator: executes directed graph of agents. Use a local lightweight orchestrator for dev and LangGraph for rich debugging; map the same graph to AWS Step Functions / Strands in production.
- Agents:
  - Planner: parse NL → ordered plan/subtasks.
  - SQLGenerator: produce candidate SQL statements (schema-aware prompts).
  - Validator: pre-execution checks (SQL safety, cost estimate, schema existence) and post-execution sanity checks.
  - SQLRunner: execute safe SQL on DuckDB (local) or submit to Athena for S3/Parquet.
  - AnalysisAgent: runs Python analysis (Pandas, statsmodels, sklearn) in a sandbox or remote job runner.
  - Summarizer: LLM-based explanation, fact-checking against results.
- LLM Adapters: pluggable adapters for OpenAI, Anthropic, Bedrock, or local models; track token usage and cost.
- Result Store & Cache: store artifacts, summaries, SQL, prompts, traces, and cached LLM outputs; use SQLite for PoC, S3/DynamoDB for production.

Data flow (typical)
1. User submits NL prompt via UI or `Agent.run()`.
2. Router selects a graph (e.g., SQL-only vs analysis) and sets execution hints (sample_only, max_rows, async).
3. Planner produces a plan; SQLGenerator produces SQL candidates.
4. Validator runs pre-checks and optionally executes sample queries.
5. SQLRunner executes SQL (sample or full) against DuckDB or Athena.
6. If analysis is requested, the AnalysisAgent runs (local sandbox for PoC or remote job for production) and returns artifacts (tables, plots, model metadata).
7. Summarizer generates narrative, checks facts against results, and writes artifacts to Result Store.
8. API returns summary and a data preview (or job id).

Routing rules (examples)
- Rule-based: keyword matching for fast routing (e.g., "regression" -> AnalysisAgent).
- Hybrid: rules first, LLM intent classifier as a fallback.
- Explicit user mode: accept `mode` flags to force `sql`, `analysis`, or `explore`.

Validation strategy
- Pre-execution: reject or require confirmation for destructive SQL, enforce row/scan limits, estimate cost.
- Post-execution: data sanity checks (nulls, types, cardinality), summary fact-checking (LLM compares summary claims to rows) and signal warnings or re-run.

Security & compliance
- Store LLM keys in Secrets Manager or environment variables for local dev.
- Use VPC endpoints and restrict egress for production; prefer Bedrock for fully in‑AWS LLM calls if compliance requires.
- Audit: persist prompts, SQL, results, and analysis code in S3 with access logs for reproducibility and NIW evidence.

Deployment variants
- Development (fast iteration):
  - LangGraph or local orchestrator + OpenAI adapter + DuckDB local + SQLite cache.
  - Streamlit UI in local container or `streamlit run`.
- Production (low baseline cost, auditable):
  - Router + Step Functions (or Strands) + Lambdas for Planner/SQLGen/Validator + SQLRunner submits to Athena + AnalysisAgent runs as Fargate job when heavy.
  - S3 for datasets and artifact storage, Secrets Manager for keys, CloudWatch/X‑Ray for traces.

Cost controls
- Default to sample-only runs; require explicit confirmation to run full scans.
- Partition and store datasets as Parquet and prefer Athena to avoid egress/hosted compute costs.
- Cache LLM outputs and query results; use cheaper models for planning and expensive models for summarization.

Observability & testing
- Capture and store per-run traces: prompts, LLM responses, generated SQL, validator logs, execution metrics, token usage.
- Tests: unit tests for router rules and agent stubs; prompt-eval suite to compare outputs across model adapters.

Implementation milestones (recommended)
1. PoC (days 0–3): scaffold package, `Agent.run` (sync), dummy LLM adapter, Router (rule-based), SQLGenerator+SQLRunner (sample mode), Validator (basic), Streamlit demo, and Jupyter magic. Use SQLite cache.
2. Agents & Orchestration (days 3–6): add AnalysisAgent with local sandboxed runner, LangGraph adapter for local visualization, and table-function UDFs for DuckDB.
3. Production wiring (days 6–10): CDK skeleton for Step Functions + Lambdas, S3 storage + Athena connector, Secrets Manager integration, sample Fargate job image for AnalysisAgent.
4. Hardening (post-PoC): add Bedrock adapter, token/cost accounting, RBAC, CI/CD, and more extensive prompt-eval tests.

Files
- `docs/architecture.md` (this file) — diagram + plan
- `src/duckagent/orchestrator/local_orchestrator.py` — implement graph runner (PoC)
- `src/duckagent/router.py` — rule-based router spec
- `infra/` — CDK skeleton for Step Functions + S3 (optional)

Next steps
- I can scaffold the PoC files now (router stub, orchestrator stub, agent stubs, Streamlit demo). Confirm and I will create the code under `~/git/duckagent/src/duckagent` and update the todo list.

Dynamic routing
----------------
The system supports dynamic routing so each request runs only the agents needed. This reduces cost, speeds responses, and improves auditability. Below is the recommended runtime behavior and interfaces.

Routing overview
- Two-stage hybrid router:
  1. Rule-based fast path: use deterministic heuristics (keyword matching, explicit `mode` flag, prompt length, presence of table/DF names) to return an intent and agent list with high confidence.
  2. LLM fallback: when rules are ambiguous, call a small/cheap LLM or local classifier to return a structured intent and execution hints.
- Planner refines the router decision when a richer plan is needed (e.g., to decide whether an existing DataFrame can be summarized directly or must be replenished from SQL).

Decision format
The router/planner should return a JSON decision object describing which agents to run and execution hints. Example:

```json
{
  "intent": "analyze",
  "confidence": 0.92,
  "agents": [
    {"name":"Planner","params":{}},
    {"name":"SQLGenerator","params":{"sample_only":true}},
    {"name":"Validator","params":{"sample_mode":true}},
    {"name":"SQLRunner","params":{"max_rows":500}},
    {"name":"AnalysisAgent","params":{"analysis_mode":"regression"}},
    {"name":"Summarizer","params":{"model":"large"}}
  ],
  "cost_estimate": {"llm_tokens":1200,"scan_bytes_est":25000000},
  "reason": "contains 'analyze' and 'drivers', implies deeper analysis"
}
```

Materializer & Tracing
----------------------

The project includes a LangGraph runtime materializer that converts Planner
decisions into runtime nodes. Execution prefers an installed SDK/runtime and
falls back to a local orchestrator when those are not available. Per-node
traces (inputs, outputs, timings, and status) are collected during execution
and redacted before any external upload. This makes it easy to visualize and
debug runs in LangSmith while keeping secrets out of telemetry.

Extension points:
- `build_runtime_graph(decision, context, agent_impls)`: materializes the
  decision into nodes/edges and accepts `agent_impls` for pluggable agents.
- `run_decision_graph(decision, context)`: attempts SDK → runtime → local
  execution and returns `node_traces` suitable for upload.


Agent capability metadata
- Each agent should expose a small capability descriptor used by the router/planner to choose cheaper alternatives first:
  - `requires_sql` (bool)
  - `can_run_on_sample` (bool)
  - `requires_sandbox` (bool)
  - `cost_tier` ("low"|"medium"|"high")
  - `estimated_time_secs` (int)

Router & Planner interfaces (suggested)
- `Router.detect_intent(prompt: str, user_mode: Optional[str], context: dict) -> {intent, confidence, hints}`
- `Planner.plan_for_intent(intent: str, prompt: str, context: dict) -> Decision`  (may call an LLM and inspect schema/context)

Execution flow
1. If the user sets an explicit `mode`, the router honors it and returns a constrained decision.
2. Router runs rule-based checks; if ambiguous, Planner runs an LLM classification to refine the plan.
3. The Orchestrator executes the returned agents in order. The Validator runs pre- and post-execution checks.
4. If a plan is costly, the API surface returns the `cost_estimate` and requires explicit confirmation to run the full job.

Validation & fallbacks
- Always run `Validator` after execution to check data sanity and summary factuality.
- If validation fails, orchestrator can: retry with samples, switch to a safer tool, or escalate to human review.

Testing & observability
- Unit tests for rule-based routing and Planner decisions.
- Prompt-eval fixtures to ensure routing stability across phrasings.
- Metrics: track routing decisions, confidence, and cost savings.
