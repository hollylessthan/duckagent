"""Simple local orchestrator to execute a decision graph of agents.

This is a synchronous, single-process runner intended for PoC and testing.
Each agent node is executed by name using small helper implementations below.
"""
from typing import Dict, Any
import pandas as pd


def _run_planner(node, state):
    # planner already ran; no-op for PoC
    return {"plan": "planner-produced-plan"}


def _run_sqlgenerator(node, state):
    # produce a dummy SQL based on prompt; real impl would call LLM
    prompt = state.get("prompt", "")
    params = node.get('params', {})
    limit = params.get('max_rows', 10)

    # If an LLM is available in state, use it to generate SQL from a template prompt
    llm = state.get("llm")
    if llm:
        p = (
            "You are an assistant that generates safe SQL for DuckDB.\n"
            f"User request: {prompt}\n"
            f"Produce a single SQL query that answers the request. Limit rows to {limit}.\n"
        )
        try:
            sql_text = llm.generate(p, max_tokens=512)
            # simple sanitization: ensure LIMIT present
            if "limit" not in sql_text.lower():
                sql_text = sql_text.strip().rstrip(";") + f" LIMIT {limit}"
            return {"sql": sql_text}
        except Exception:
            # fallback to simple SQL
            pass

    # Prefer using a registered DataFrame table name if available in state
    table_name = state.get("full_df_table_name")
    # If not explicitly provided, try to discover an existing table on the
    # connection (useful when callers registered a DataFrame manually).
    if not table_name:
        conn = state.get("conn")
        if conn is not None:
            try:
                cur = conn.execute("SHOW TABLES")
                try:
                    df_tables = cur.fetchdf()
                    if not df_tables.empty:
                        # duckdb SHOW TABLES returns a column named 'name' or similar;
                        # fallback to first column value for the first row.
                        col = df_tables.columns[0]
                        table_name = str(df_tables.iloc[0][col])
                except Exception:
                    rows = cur.fetchall()
                    if rows:
                        # rows may be list of tuples like [("table_name",), ...]
                        table_name = str(rows[0][0])
            except Exception:
                # if SHOW TABLES fails (non-duckdb conn), ignore and fall back
                table_name = None

    if not table_name:
        table_name = "sample_table"

    sql = f"SELECT * FROM {table_name} LIMIT {limit}"
    return {"sql": sql}


def _run_validator(node, state):
    # basic checks for PoC
    return {"valid": True, "issues": []}


def _run_sqlrunner(node, state):
    conn = state.get("conn")
    sql = state.get("last_sql")
    rows = []
    df = None
    if conn is not None and sql:
        try:
            # duckdb.Cursor returns a pandas DataFrame via fetchdf
            cur = conn.execute(sql)
            try:
                df = cur.fetchdf()
                rows = df.head(20).to_dict(orient="records")
            except Exception:
                rows = cur.fetchall()
        except Exception as e:
            return {"error": str(e)}
    else:
        # return a fake preview
        rows = [{"col1": 1, "col2": "example"}]
    return {"rows_preview": rows, "full_df": df}


def _run_analysisagent(node, state):
    # PoC: run a tiny pandas analysis on full_df if present
    df = state.get("full_df")
    result = {"analysis_code": None, "artifacts": {}, "metrics": {}}
    if isinstance(df, pd.DataFrame):
        result["artifacts"]["describe"] = df.describe().to_dict()
        result["metrics"]["n_rows"] = len(df)
    else:
        result["artifacts"]["note"] = "no dataframe provided to AnalysisAgent"
    return result


def _run_summarizer(node, state):
    # PoC summarizer: prefer operating on a provided full DataFrame when available.
    plan = state.get("plan_text", "(no plan)")
    last_sql = state.get("last_sql", "(no sql)")
    rows = state.get("rows_preview", [])
    full_df = state.get("full_df")

    llm = state.get("llm")
    # If a full DataFrame is present prefer summarizing it directly
    if full_df is not None:
        try:
            import pandas as pd

            if isinstance(full_df, pd.DataFrame):
                n_rows, n_cols = full_df.shape
                cols = list(full_df.columns)
                sample_rows = full_df.head(5).to_dict(orient="records")
                if llm:
                    prompt = (
                        "You are an assistant that summarizes a pandas DataFrame.\n"
                        f"Plan: {plan}\n"
                        f"DataFrame shape: {n_rows} rows x {n_cols} cols\n"
                        f"Columns: {cols}\n"
                        f"Sample rows: {sample_rows}\n"
                        "Produce a concise human-readable summary (3-5 sentences)."
                    )
                    try:
                        text = llm.generate(prompt, max_tokens=300)
                        return {"summary": text, "llm_used": True}
                    except Exception:
                        # fall back to non-LLM summary if generation fails
                        pass

                # fallback non-LLM summary
                summary = (
                    f"DataFrame with {n_rows} rows and {n_cols} columns. Columns: {cols}. "
                    f"Sample rows: {sample_rows}"
                )
                return {"summary": summary}
        except Exception:
            # if pandas isn't available or summarization fails, fall through
            pass

    # No full_df present: fall back to summarizing the SQL/rows preview
    if llm:
        sample_text = str(rows[:5])
        prompt = (
            "You are an assistant that summarizes analysis findings.\n"
            f"Plan: {plan}\n"
            f"SQL: {last_sql}\n"
            f"Sample rows: {sample_text}\n"
            "Produce a concise human-readable summary (3-5 sentences)."
        )
        try:
            text = llm.generate(prompt, max_tokens=300)
            return {"summary": text, "llm_used": True}
        except Exception:
            pass

    # Helpful fallback message when there's no data to summarize
    if not rows:
        message = (
            "No data available to summarize. Provide a DataFrame via ``Agent.run(..., "
            "data=...)`` or run a SQL-producing plan so results can be summarized."
        )
        return {"summary": message, "llm_used": False}

    summary = f"Plan: {plan}\nSQL: {last_sql}\nRows preview count: {len(rows)}"
    return {"summary": summary, "llm_used": False}


AGENT_IMPL = {
    "Planner": _run_planner,
    "SQLGenerator": _run_sqlgenerator,
    "Validator": _run_validator,
    "SQLRunner": _run_sqlrunner,
    "AnalysisAgent": _run_analysisagent,
    "Summarizer": _run_summarizer,
}


def execute(decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a decision dict produced by Planner and return a result dict.

    The context dictionary contains runtime objects like `conn`, `prompt`.
    """
    # Seed runtime state from the provided context so agents can access
    # objects like the active DuckDB connection, an LLM wrapper, and an
    # explicit full DataFrame passed via `Agent.run(..., data=...)`.
    state = {
        "conn": context.get("conn"),
        "prompt": context.get("prompt"),
        "llm": context.get("llm"),
        "full_df": context.get("full_df"),
    }

    # Respect explicit preference to use LangGraph: only attempt a LangGraph
    # runtime run when the caller set `context['prefer_langgraph'] = True`.
    # This avoids surprising graph execution when callers prefer local runs.
    prefer_lg = context.get("prefer_langgraph", False)
    if prefer_lg:
        try:
            import langgraph  # type: ignore

            if hasattr(langgraph, "Graph"):
                try:
                    lg = langgraph
                    graph = lg.Graph()
                    nodes = {}
                    for idx, node in enumerate(decision.get("agents", [])):
                        name = node.get("name")

                        def make_callable(n):
                            def fn(state_inner, params=n.get("params", {})):
                                one_decision = {"agents": [n]}
                                return execute(one_decision, state_inner)

                            return fn

                        node_id = f"node_{idx}_{name}"
                        nodes[node_id] = graph.add_node(name=node_id, fn=make_callable(node))

                    node_ids = list(nodes.keys())
                    for a, b in zip(node_ids, node_ids[1:]):
                        graph.add_edge(nodes[a], nodes[b])

                    run_result = graph.run(state)
                    return {"langgraph_result": run_result}
                except Exception:
                    # If LangGraph runtime fails, log and fall back to local execution
                    import logging

                    logging.getLogger(__name__).exception("LangGraph runtime failed; falling back to local orchestrator")
        except Exception:
            # LangGraph not installed or import failed; proceed with local execution
            pass
    results = {}
    for raw_node in decision.get("agents", []):
        # normalize node: accept either a string name or a dict {name, params}
        if isinstance(raw_node, str):
            node = {"name": raw_node, "params": {}}
        else:
            node = raw_node or {"name": "", "params": {}}

        name = node.get("name")
        params = node.get("params", {})
        impl = AGENT_IMPL.get(name)
        if not impl:
            results[name] = {"error": "unknown agent"}
            continue
        # make node accessible to impl
        node_with_params = {"params": params}
        out = impl(node_with_params, state)
        results[name] = out
        # side-effectful state updates
        if isinstance(out, dict):
            if "sql" in out:
                state["last_sql"] = out["sql"]
            if "rows_preview" in out:
                state["rows_preview"] = out["rows_preview"]
            if "full_df" in out:
                state["full_df"] = out["full_df"]
            if "plan" in out:
                state["plan_text"] = out["plan"]

    # build top-level result
    top = {
        "decision": decision,
        "results": results,
        "summary": state.get("rows_preview", [])[:5],
    }
    # if summarizer produced text include it
    summ = results.get("Summarizer")
    if summ and isinstance(summ, dict) and "summary" in summ:
        top["summary_text"] = summ["summary"]
    return top
