# duckagent

Install
-------

Install runtime dependencies and the package (use your project virtualenv):

```bash
pip install -r requirements.txt
pip install -e .
```

Quickstart
-----------
- `Agent.run(prompt)` runs the router/planner/orchestrator pipeline synchronously and returns a structured result.
- You can pass a Pandas `DataFrame` directly to `Agent.run(..., data=...)`. When provided the Planner will prefer
  a Summarizer-only plan and the Summarizer will operate on the DataFrame directly.
- If you pass a DuckDB `conn` to `Agent(conn=...)` and the connection supports `register(...)`, the DataFrame
  will be registered as a SQL table under a configurable `table_name` (default `full_df`). This allows SQL
  generation and ad-hoc queries to reference the provided dataset.
----------
Example
-----------

- **Run programmatically:**

```python
from duckagent.agent import Agent

agent = Agent()
res = agent.run("Summarize revenue by country")
print(res)
```

- **Pass a DataFrame to the agent:**

```python
import pandas as pd
from duckagent.agent import Agent

df = pd.DataFrame({"country": ["US","CA"], "revenue": [100,200]})
agent = Agent()
res = agent.run("Summarize revenue by country", data=df)
```

- **Notebook:** load the IPython extension and run `%%duckagent` in a cell.

Example notebook usage:

1. In a notebook cell: `%load_ext duckagent.ipython_magic`
2. Create a DataFrame named `sales_df`.
3. Run:

```python
%%duckagent --df=sales_df --as_table=agent_result
Summarize revenue by country
```

If `--as_table` is used and the magic creates a DuckDB connection, the magic will expose the connection as
`_duckagent_conn` and register the result table for further SQL queries.

More details on architecture, design decisions, and extension ideas are in `docs/architecture.md`.

Contributing & tests
---------------------

Run the tests with:

```bash
./scripts/run_tests.sh
```

For developer setup, see `pyproject.toml` and `requirements.txt`.


Further reading
---------------
See `docs/architecture.md` for details on the orchestration graph, routing rules, and deployment recommendations.

Environment variables
---------------------

To enable LangSmith integrations or other hosted tooling, set the `LANGSMITH_API_KEY` environment variable in your shell:

Temporary (current shell):

```bash
export LANGSMITH_API_KEY="sk-..."
```

Persistent (every new shell): add the same line to your `~/.zshrc` or `~/.zprofile`.

Alternatively you can create a local `.env` file (not committed) and load it via `direnv` or `python-dotenv`. See `.env.example` for the expected variable name.

LangGraph runtime & LangSmith tracing
------------------------------------
- The adapter materializes Planner decisions into a runtime graph and prefers runtime/SDK execution in this order:
  1. `langgraph_sdk` (if installed)
  2. `langgraph` runtime (if installed)
  3. Local orchestrator fallback.
- Per-node traces (inputs/outputs/timings) are collected and redacted before any upload.
- Optional dependencies: `langgraph`, `langgraph_sdk`, and `langsmith`. To enable LangSmith upload set `LANGSMITH_API_KEY` in your environment.
- Run the supervisor demo (uses `MockLLM` when `OPENAI_API_KEY` is not set):

```bash
export LANGSMITH_API_KEY="sk-..."   # optional
python3 examples/langsmith_supervisor_full_demo.py
```
