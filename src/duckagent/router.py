"""Rule-based router with a lightweight interface.
"""
from typing import Optional, Dict, Any
import re


class Router:
    def __init__(self, rules: Optional[Dict[str, Any]] = None):
        # rules can be extended; kept simple for PoC
        self.rules = rules or {}

    def detect_intent(self, prompt: str, user_mode: Optional[str] = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Return a decision dict with intent, confidence, suggested agents and hints.

        This is a fast rule-based detector. For ambiguous cases, confidence is lower
        and the planner (LLM) can be invoked to refine the plan.
        """
        text = (prompt or "").lower()
        # explicit override
        if user_mode:
            return {"intent": user_mode, "confidence": 0.95, "agents": [], "hints": {}}

        # use regex word-boundary matching to avoid substring collisions
        analyze_pattern = re.compile(r"\b(analy|analysis|analyze|regress|correlat|drivers|model|predict)\b", re.I)
        summarize_pattern = re.compile(r"\b(summary|summarize|summarise|describe|overview|insight)\b", re.I)
        sql_pattern = re.compile(r"\b(count|how many|top|sum|avg|group by|order by|select|min|max)\b", re.I)

        # check analysis first
        if analyze_pattern.search(text):
            return {
                "intent": "analyze",
                "confidence": 0.92,
                "agents": ["Planner", "SQLGenerator", "Validator", "SQLRunner", "AnalysisAgent", "Summarizer"],
                "hints": {"sample_only": True},
            }
        # check summarize before sql to avoid false positives (e.g., 'summary' contains 'sum')
        if summarize_pattern.search(text):
            # summarization often can run directly on an in-memory dataframe
            return {
                "intent": "summarize",
                "confidence": 0.93,
                "agents": ["Planner", "Summarizer"],
                "hints": {"use_existing_df": True},
            }

        if sql_pattern.search(text):
            return {
                "intent": "sql",
                "confidence": 0.9,
                "agents": ["Planner", "SQLGenerator", "Validator", "SQLRunner", "Summarizer"],
                "hints": {"sample_only": True},
            }

        # fallback: low confidence, ask planner to disambiguate
        return {"intent": "unknown", "confidence": 0.5, "agents": [], "hints": {}}
