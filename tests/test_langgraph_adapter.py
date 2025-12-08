import pytest

from duckagent.adapters.langgraph_adapter import LangGraphAdapter, LangGraphUnavailable


def test_decision_to_graph_basic():
    adapter = LangGraphAdapter()
    decision = {
        "intent": "analyze",
        "agents": [
            {"name": "Planner", "params": {}},
            {"name": "SQLGenerator", "params": {"sample_only": True}},
            {"name": "Summarizer", "params": {"model": "small"}},
        ],
    }

    graph = adapter.decision_to_graph(decision)
    assert "nodes" in graph and "edges" in graph
    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 2


def test_graph_to_execution_result_basic():
    adapter = LangGraphAdapter()
    fake_execution = {
        "status": "completed",
        "nodes": {
            "Planner": {"status": "ok", "output": "planner output"},
            "SQLGenerator": {"status": "ok", "output": "SELECT 1"},
            "Summarizer": {"status": "ok", "output": "Summary text"},
        },
        "metrics": {"token_count": 10},
        "trace_id": "trace-123",
    }

    result = adapter.graph_to_execution_result(fake_execution)
    assert result["execution"]["status"] == "completed"
    assert "Summarizer" in result["execution"]["results"]


def test_execute_graph_without_client_raises():
    adapter = LangGraphAdapter()
    with pytest.raises(LangGraphUnavailable):
        adapter.execute_graph({})
