# duckagent PoC

Lightweight PoC for a DuckDB-integrated multi-agent analytics assistant.

Quick usage
-----------

- `Agent.run(prompt)` runs the router/planner/orchestrator pipeline synchronously and returns a structured result.
- You can pass a Pandas `DataFrame` directly to `Agent.run(..., data=...)`. When provided the Planner will prefer
  a Summarizer-only plan and the Summarizer will operate on the DataFrame directly.
- If you pass a DuckDB `conn` to `Agent(conn=...)` and the connection supports `register(...)`, the DataFrame
  will be registered as a SQL table under a configurable `table_name` (default `full_df`). This allows SQL
  generation and ad-hoc queries to reference the provided dataset.

Example
-------

```python
import pandas as pd
import duckdb
from duckagent.agent import Agent

# DataFrame
df = pd.DataFrame({"country": ["US","CA"], "revenue": [100,200]})

# Summarize directly from Python
agent = Agent(conn=None)
res = agent.run("Summarize revenue by country", data=df)
print(res["execution"]["results"].get("Summarizer"))

# Or register DataFrame into DuckDB under a custom table name
conn = duckdb.connect(':memory:')
agent_db = Agent(conn=conn, table_name="my_table")
res_db = agent_db.run("Summarize revenue by country", data=df)
print(conn.execute("SELECT * FROM my_table").fetchdf())
```

Further reading
---------------
See `docs/architecture.md` for details on the orchestration graph, routing rules, and deployment recommendations.
