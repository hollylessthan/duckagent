"""Agent façade that wires router, planner and orchestrator for a simple run API.
"""
from typing import Optional, Dict, Any
from .router import Router
from .planner import Planner
from .orchestrator import execute as orch_execute


class Agent:
    def __init__(self, conn: Optional[Any] = None, llm: Optional[Any] = None, cache: Optional[Any] = None):
        self.conn = conn
        self.llm = llm
        self.cache = cache
        self.router = Router()
        self.planner = Planner(llm=llm)

    def run(self, prompt: str, mode: Optional[str] = None, context: Optional[Dict[str, Any]] = None, data: Optional[Any] = None) -> Dict[str, Any]:
        """Run a natural-language prompt through the routing/planning/orchestration pipeline.

        Returns a structured result dictionary for quick consumption in notebooks or Streamlit.
        """
        ctx = context or {}
        # add common runtime objects to context, including llm adapter if present
        ctx.update({"conn": self.conn, "prompt": prompt, "llm": self.llm})

        # If explicit data was provided prefer it: inject into context as `full_df` and
        # register it on the DuckDB connection when possible. This explicit param
        # overrides any implicitly-detected data in `context`.
        if data is not None:
            ctx["full_df"] = data
            try:
                # duckdb.Connection.register is a convenient way to expose a pandas
                # DataFrame as a SQL table to queries executed on the connection.
                conn = ctx.get("conn")
                if conn is not None and hasattr(conn, "register"):
                    # register under a well-known name so SQL generators can target it
                    conn.register("full_df", data)
                    ctx["full_df_table_name"] = "full_df"
            except Exception:
                # ignore registration errors for PoC
                pass

        # Stage 1: router
        decision = self.router.detect_intent(prompt, user_mode=mode, context=ctx)

        # If router is unsure, ask planner for a concrete decision
        if decision.get("confidence", 0) < 0.7 or not decision.get("agents"):
            decision = self.planner.plan_for_intent(decision.get("intent", "unknown"), prompt, ctx)

        # If planner returned a decision object keep it
        if "agents" not in decision:
            # fallback to planner
            decision = self.planner.plan_for_intent(decision.get("intent", "unknown"), prompt, ctx)

        # Execute the decision graph
        out = orch_execute(decision, ctx)

        # Attach some metadata
        result = {
            "decision": decision,
            "execution": out,
            "metadata": {"router_confidence": decision.get("confidence", None)},
        }
        return result
