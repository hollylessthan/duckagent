import pandas as pd
from duckagent.agent import Agent


def test_use_langgraph_true_calls_adapter(monkeypatch):
    called = {}

    def fake_run(decision, ctx):
        called['called'] = True
        return {'fake': 'ok'}

    monkeypatch.setattr('duckagent.adapters.langgraph_adapter.run_decision_graph', fake_run)

    agent = Agent(use_langgraph=True)
    res = agent.run("Please summarize this dataset", data=pd.DataFrame({"a": [1, 2]}))

    assert called.get('called') is True
    assert res.get('execution') == {'fake': 'ok'}


def test_use_langgraph_false_does_not_call_adapter(monkeypatch):
    called = {}

    def fake_run(decision, ctx):
        called['called'] = True
        return {'fake': 'ok'}

    monkeypatch.setattr('duckagent.adapters.langgraph_adapter.run_decision_graph', fake_run)

    agent = Agent(use_langgraph=False)
    res = agent.run("Please summarize this dataset", data=pd.DataFrame({"a": [1, 2]}))

    # adapter should not have been called
    assert called.get('called') is None
    # ensure we still get an execution result (fallback to local orchestrator)
    assert isinstance(res.get('execution'), dict)


def test_use_langgraph_auto_honors_HAS_LANGGRAPH(monkeypatch):
    called = {}

    def fake_run(decision, ctx):
        called['called'] = True
        return {'fake_auto': 'ok'}

    monkeypatch.setattr('duckagent.adapters.langgraph_adapter.run_decision_graph', fake_run)
    # set HAS_LANGGRAPH on the module (use full attribute path)
    monkeypatch.setattr('duckagent.adapters.langgraph_adapter.HAS_LANGGRAPH', True, raising=False)

    agent = Agent(use_langgraph='auto')
    res = agent.run("Please summarize this dataset", data=pd.DataFrame({"a": [1, 2]}))

    assert called.get('called') is True
    assert res.get('execution') == {'fake_auto': 'ok'}
