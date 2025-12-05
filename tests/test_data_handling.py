import pandas as pd
from duckagent.agent import Agent
from duckagent.planner import Planner
from duckagent.orchestrator import execute as orch_execute


def test_agent_run_with_data_param():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    agent = Agent(conn=None)
    prompt = "Summarize the provided dataset"
    res = agent.run(prompt, data=df)
    # planner should detect data and choose Summarizer-only
    agents = res["decision"]["agents"]
    def _is_summarizer(n):
        if isinstance(n, str):
            return n == "Summarizer"
        return n.get("name") == "Summarizer"

    assert any(_is_summarizer(n) for n in agents)
    # execution should include a Summarizer result with a summary string mentioning rows
    summ = res["execution"]["results"].get("Summarizer")
    assert isinstance(summ, dict)
    assert "summary" in summ
    assert "rows" in summ["summary"] or "DataFrame" in summ["summary"]


def test_planner_detects_context_df():
    df = pd.DataFrame({"x": [0, 1]})
    planner = Planner()
    decision = planner.plan_for_intent("summarize", "please summarize", {"full_df": df})
    assert decision["agents"][0]["name"] == "Summarizer"
    assert decision["hints"].get("use_existing_df") is True


def test_summarizer_no_data_message():
    # execute a decision with only Summarizer and no data/rows
    decision = {"agents": [{"name": "Summarizer", "params": {}}]}
    ctx = {"prompt": "Give me a summary"}
    out = orch_execute(decision, ctx)
    summ = out["results"].get("Summarizer")
    assert isinstance(summ, dict)
    assert "summary" in summ
    assert "No data available to summarize" in summ["summary"]
