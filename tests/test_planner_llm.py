import json

from duckagent.planner import Planner
from duckagent.llm_adapter import MockLLM


def test_planner_with_mock_llm_returns_valid_decision():
    mock = MockLLM()

    # craft a mock JSON decision the LLM would return
    decision = {
        "intent": "analyze",
        "agents": [
            {"name": "SQLGenerator", "params": {"sample_only": True}},
            {"name": "SQLRunner", "params": {"max_rows": 50}},
            {"name": "Summarizer", "params": {}},
        ],
        "hints": {"sample_only": True},
    }

    # MockLLM.generate returns a string; we'll wrap the JSON as the mock output
    def gen(prompt, **opts):
        return json.dumps(decision)

    mock.generate = gen

    planner = Planner(llm=mock)
    ctx = {}
    res = planner.plan_for_intent("analyze", "Find trends in sales", ctx)

    assert isinstance(res, dict)
    assert res["intent"] == "analyze"
    assert isinstance(res["agents"], list)
    names = [a["name"] for a in res["agents"]]
    assert "SQLGenerator" in names
    assert "SQLRunner" in names


def test_planner_falls_back_on_invalid_json():
    mock = MockLLM()

    # return invalid JSON
    mock.generate = lambda prompt, **opts: "I think we should do X and Y"

    planner = Planner(llm=mock)
    res = planner.plan_for_intent("summarize", "Give summary", {})
    assert isinstance(res, dict)
    # fallback plan for summarize contains Summarizer
    names = [a["name"] for a in res["agents"]]
    assert "Summarizer" in names
