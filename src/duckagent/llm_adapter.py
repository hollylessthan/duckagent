"""LLM adapter layer with OpenAI implementation and a simple Mock fallback.

Provides a small unified interface used by Planner/SQLGenerator/Summarizer.
"""
from typing import List, Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)


class BaseLLM:
    def chat(self, messages: List[Dict[str, str]], **opts) -> Dict[str, Any]:
        raise NotImplementedError()

    def generate(self, prompt: str, **opts) -> str:
        raise NotImplementedError()


class OpenAIAdapter(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        try:
            import openai
        except Exception:
            raise
        self.openai = openai
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if self.api_key:
            self.openai.api_key = self.api_key

    def chat(self, messages: List[Dict[str, str]], **opts) -> Dict[str, Any]:
        # Use ChatCompletion if available
        try:
            resp = self.openai.ChatCompletion.create(model=self.model, messages=messages, **opts)
            text = resp.choices[0].message.content
            return {"text": text, "raw": resp}
        except Exception as e:
            logger.exception("OpenAI chat error: %s", e)
            raise

    def generate(self, prompt: str, **opts) -> str:
        messages = [{"role": "user", "content": prompt}]
        out = self.chat(messages, **opts)
        return out.get("text", "")


class MockLLM(BaseLLM):
    def chat(self, messages: List[Dict[str, str]], **opts) -> Dict[str, Any]:
        # naive echo-like behavior for PoC
        last = messages[-1]["content"] if messages else ""
        return {"text": f"[MOCK LLM] {last}", "raw": None}

    def generate(self, prompt: str, **opts) -> str:
        return f"[MOCK LLM] Generated response for prompt: {prompt[:200]}"
