"""Planner that can optionally use an LLM to produce a validated decision.

By default the Planner behaves as a simple deterministic mapper (PoC). When an
LLM adapter is provided, Planner will attempt to ask the LLM to emit a JSON
decision object. The returned JSON is parsed and validated against a small
allowlist of agent names and param types. On failure the planner falls back to
the safe deterministic plan.
"""
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


DEFAULT_AGENT_NAMES = [
    "Planner",
    "SQLGenerator",
    "Validator",
    "SQLRunner",
    "AnalysisAgent",
    "Summarizer",
]


class Planner:
    def __init__(self, llm=None):
        # llm is a pluggable adapter; PoC uses no external LLM
        self.llm = llm

    def _default_plan(self, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        decision = {
            "intent": intent,
            "agents": [],
            "hints": {},
            "reason": "planner default mapping",
        }

        full_df = context.get("full_df")
        rows_preview = context.get("rows_preview")
        if full_df is not None or (rows_preview and len(rows_preview) > 0):
            decision["agents"] = [{"name": "Summarizer", "params": {}}]
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
            decision["agents"] = [
                {"name": "Planner", "params": {}},
                {"name": "Summarizer", "params": {}},
            ]

        decision["cost_estimate"] = {"llm_tokens": 100, "scan_bytes_est": 0}
        return decision

    def _validate_decision(self, obj: Any) -> Dict[str, Any]:
        """Validate and normalize a decision object parsed from LLM output.

        Ensures allowed agent names and simple param types (dict/primitive).
        Raises ValueError on invalid structure.
        """
        if not isinstance(obj, dict):
            raise ValueError("decision must be a JSON object")
        agents = obj.get("agents")
        if agents is None or not isinstance(agents, list):
            raise ValueError("decision.agents must be a list")

        normalized = {"intent": obj.get("intent"), "agents": [], "hints": obj.get("hints", {})}
        for a in agents:
            if not isinstance(a, dict):
                raise ValueError("each agent must be an object")
            name = a.get("name")
            if name not in DEFAULT_AGENT_NAMES:
                raise ValueError(f"agent name '{name}' is not allowed")
            params = a.get("params", {}) or {}
            if not isinstance(params, dict):
                raise ValueError("agent.params must be an object/dict")
            # shallow sanitize params: only allow simple JSON types
            safe_params = {}
            for k, v in params.items():
                if isinstance(v, (str, int, float, bool, type(None), list, dict)):
                    safe_params[k] = v
            normalized["agents"].append({"name": name, "params": safe_params})
        return normalized

    def plan_for_intent(self, intent: str, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return a Decision dict describing agents (with params) and execution hints.

        If an LLM is provided, attempt to generate a JSON decision from it and
        validate the result. On any failure fall back to a safe deterministic
        plan.
        """
        # Attempt quick deterministic shortcut for existing data
        full_df = context.get("full_df")
        rows_preview = context.get("rows_preview")
        if full_df is not None or (rows_preview and len(rows_preview) > 0):
            return self._default_plan(intent, context)

        # If no LLM available, return default plan
        if not self.llm:
            return self._default_plan(intent, context)

        # Build a safe prompt asking for a JSON decision
        prompt_template = (
            "You are a planner that returns a JSON object describing an execution plan. "
            "Respond ONLY with valid JSON. The JSON must contain an 'intent' string and an 'agents' list. "
            "Each agent in 'agents' must be an object with 'name' (one of: "
            + ", ".join(DEFAULT_AGENT_NAMES)
            + ") and optional 'params' (a simple JSON object).\n"
            "User prompt: "
            f"{prompt}\n"
            "Produce the JSON decision now."
        )

        try:
            raw = self.llm.generate(prompt_template, max_tokens=512)
            # Try to locate JSON in the output (allow surrounding text)
            text = raw.strip()
            # If the model included markdown or backticks, strip them
            if text.startswith("```") and text.endswith("```"):
                # remove triple-backticks wrapper
                text = "\n".join(text.split("\n")[1:-1])
            # Find first '{' to be robust to leading commentary
            start = text.find("{")
            if start != -1:
                text = text[start:]
            obj = json.loads(text)
            validated = self._validate_decision(obj)
            # Ensure intent is set
            if not validated.get("intent"):
                validated["intent"] = intent
            return validated
        except Exception as e:
            logger.exception("LLM planner failed or returned invalid plan: %s", e)
            return self._default_plan(intent, context)
