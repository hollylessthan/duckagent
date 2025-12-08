import pytest

from duckagent.adapters.langgraph_adapter import LangGraphAdapter, LangGraphAdapterError


def test_send_to_langsmith_raises_when_sdk_missing(monkeypatch):
    adapter = LangGraphAdapter()
    graph = {"nodes": [], "edges": [], "metadata": {}}

    # Ensure langsmith is not importable in this test environment
    monkeypatch.setitem(__import__('sys').modules, 'langsmith', None)

    with pytest.raises(LangGraphAdapterError):
        adapter.send_to_langsmith(graph)
