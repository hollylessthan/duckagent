"""IPython extension providing a `%%duckagent` cell magic.

Usage in a notebook:

%load_ext duckagent.ipython_magic
%%duckagent [--df=df_variable] [--conn_var=conn_variable] [--table_name=name]
Analyze retention drop last quarter

The magic will:
 - optionally locate a pandas DataFrame in the notebook namespace and register
   it into a DuckDB connection
 - create or reuse a DuckDB in-memory connection
 - call `Agent.run(...)` and pretty-print the `execution` result
"""
from IPython.core.magic import register_cell_magic
from IPython import get_ipython

def _find_in_user_ns(name):
    ip = get_ipython()
    return ip.user_ns.get(name, None)


@register_cell_magic
def duckagent(line, cell):
    """Cell magic to run duckagent against the notebook cell text.

    Simple flags supported in `line`:
      --df=<varname>         # name of a pandas DataFrame in notebook namespace
      --conn_var=<varname>   # name of a duckdb connection variable to reuse
      --table_name=<name>    # table name to register the DataFrame as
    """
    # lazy imports
    import shlex
    import duckdb
    import pandas as pd
    from duckagent.agent import Agent

    args = shlex.split(line)
    opts = {}
    for a in args:
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            opts[k] = v

    df_var = opts.get("df")
    conn_var = opts.get("conn_var")
    table_name = opts.get("table_name", "full_df")

    # get user objects
    df = _find_in_user_ns(df_var) if df_var else None
    conn = _find_in_user_ns(conn_var) if conn_var else None

    created_conn = False
    if conn is None:
        conn = duckdb.connect(":memory:")
        created_conn = True

    # if DataFrame present, register it on the connection
    if df is not None and hasattr(conn, "register"):
        try:
            conn.register(table_name, df)
        except Exception:
            pass

    # create Agent with the connection
    agent = Agent(conn=conn, table_name=table_name)

    prompt = cell.strip()
    print(f"Running duckagent prompt: {prompt!r}")

    # support --as_table=<name> to register the agent output as a SQL table
    as_table = opts.get("as_table")

    # run the agent
    res = agent.run(prompt, data=df)

    # execution payload (dict or other)
    execution = res.get("execution") if isinstance(res, dict) else res

    # imports for pretty display
    import json
    from IPython.display import display, HTML

    # Try to locate a pandas DataFrame in the execution results
    def _find_dataframe_from_execution(execution_obj):
        try:
            results = execution_obj.get("results") if isinstance(execution_obj, dict) else None
            if isinstance(results, dict):
                for v in results.values():
                    try:
                        if isinstance(v, pd.DataFrame):
                            return v
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            if isinstance(execution_obj, dict):
                for v in execution_obj.values():
                    if isinstance(v, pd.DataFrame):
                        return v
                    if isinstance(v, dict):
                        for sv in v.values():
                            if isinstance(sv, pd.DataFrame):
                                return sv
        except Exception:
            pass
        return None

    df_out = _find_dataframe_from_execution(execution)

    # If user requested to register as a table and a DataFrame was found,
    # attempt to register it on the conn (best-effort).
    registered_table = None
    if as_table and df_out is not None:
        try:
            if hasattr(conn, "register"):
                conn.register(as_table, df_out)
                registered_table = as_table
        except Exception:
            try:
                conn.execute(f"CREATE TABLE {as_table} AS SELECT * FROM df_out")
                registered_table = as_table
            except Exception:
                registered_table = None

    # Pretty display: DataFrame if available, else show execution summary or trace
    if df_out is not None:
        display(df_out)
        if registered_table:
            display(HTML(f"<div>Registered result as table: <b>{registered_table}</b></div>"))
    else:
        summary_text = None
        if isinstance(execution, dict):
            summary_text = execution.get("summary_text") or execution.get("summary")
            if isinstance(summary_text, list):
                summary_text = "\n".join(map(str, summary_text))

        if summary_text:
            display(HTML(f"<pre>{summary_text}</pre>"))
        else:
            try:
                pretty = json.dumps(execution, indent=2, default=str)
            except Exception:
                pretty = str(execution)
            html = f"<details><summary>Execution trace (click to expand)</summary><pre>{pretty}</pre></details>"
            display(HTML(html))

    # If we created the connection and the result was registered as a table,
    # expose the connection in the notebook namespace so users can query it.
    if created_conn and registered_table:
        try:
            ip = get_ipython()
            ip.user_ns.setdefault("_duckagent_conn", conn)
            from IPython.display import HTML as _HTML
            display(_HTML(f"<div>DuckDB connection stored as variable <code>_duckagent_conn</code>. Query: SELECT * FROM {registered_table}</div>"))
        except Exception:
            pass
    else:
        # cleanup if we created the connection
        if created_conn:
            try:
                conn.close()
            except Exception:
                pass


def load_ipython_extension(ipython):
    """IPython extension entrypoint. Register the cell magic when loaded."""
    # The `@register_cell_magic` decorator registers the magic on import, but
    # keep this hook for explicit extension loading semantics.
    pass


def unload_ipython_extension(ipython):
    # no-op for now
    pass
