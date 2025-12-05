"""Planner stub that refines routing decisions into an executable graph/decision.
"""
from typing import Dict, Any


class Planner:
    def __init__(self, llm=None):
        # llm is a pluggable adapter; PoC uses no external LLM
        self.llm = llm

    def plan_for_intent(self, intent: str, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return a Decision dict describing agents (with params) and execution hints.

        For PoC this returns a deterministic mapping; in the future this will call an LLM
        to produce a richer plan.
        """
        decision = {
            "intent": intent,
            "agents": [],
            "hints": {},
            "reason": "planner default mapping",
        }

        # If the runtime context already contains a DataFrame or a rows preview,
        # prefer a Summarizer-only plan so we operate on the provided data instead
        # of trying to re-generate SQL. This supports the explicit `data` param
        # passed to `Agent.run(...)` which injects `full_df` into `context`.
        full_df = context.get("full_df")
        rows_preview = context.get("rows_preview")
        if full_df is not None or (rows_preview and len(rows_preview) > 0):
            decision["agents"] = [
                {"name": "Summarizer", "params": {}},
            ]
            decision["hints"] = {"use_existing_df": True}
            decision["reason"] = "context contains data; prefer summarizer"
            decision["cost_estimate"] = {"llm_tokens": 20, "scan_bytes_est": 0}
            return decision

        if intent == "analyze":
            decision["agents"] = [
                {"name": "Planner", "params": {}},
                {"name": "SQLGenerator", "params": {"sample_only": True}},
                {"name": "Validator", "params": {"sample_mode": True}},
                {"name": "SQLRunner", "params": {"max_rows": 1000}},
                {"name": "AnalysisAgent", "params": {"analysis_mode": "regression"}},
                {"name": "Summarizer", "params": {"model": "default"}},
            ]
            decision["hints"] = {"confirm_full_run": True}
        elif intent == "sql":
            decision["agents"] = [
                {"name": "Planner", "params": {}},
                {"name": "SQLGenerator", "params": {"sample_only": True}},
                {"name": "Validator", "params": {}},
                {"name": "SQLRunner", "params": {"max_rows": 500}},
                {"name": "Summarizer", "params": {}},
            ]
        elif intent == "summarize":
            decision["agents"] = [
                {"name": "Planner", "params": {}},
                {"name": "Summarizer", "params": {}},
            ]
        else:
            # unknown -> default to planner + summarizer to be safe
            decision["agents"] = [
                {"name": "Planner", "params": {}},
                {"name": "Summarizer", "params": {}},
            ]

        # add a lightweight cost estimate (PoC)
        decision["cost_estimate"] = {"llm_tokens": 100, "scan_bytes_est": 0}
        return decision
