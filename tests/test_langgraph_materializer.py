import re
from duckagent.adapters.langgraph_adapter import build_runtime_graph, run_decision_graph, _redact


def test_build_runtime_graph_basic():
    decision = {
        "intent": "test_intent",
        "agents": [
            {"name": "AgentA", "params": {"sample_only": True}},
            {"name": "AgentB", "params": {"api_key": "sk-ABCDEF123456"}},
        ],
    }

    mat = build_runtime_graph(decision, {})
    assert "graph_dict" in mat
    graph = mat["graph_dict"]
    assert isinstance(graph.get("nodes"), list)
    assert len(graph["nodes"]) == 2
    assert graph["nodes"][0]["name"] == "AgentA"
    assert graph["nodes"][1]["name"] == "AgentB"
    # edges should connect nodes linearly
    assert graph.get("edges") and len(graph["edges"]) == 1
    runtime_nodes = mat["runtime_nodes"]
    assert len(runtime_nodes) == 2
    assert callable(runtime_nodes[0]["fn"]) and callable(runtime_nodes[1]["fn"])


def test_run_decision_graph_traces_and_redaction():
    # Decision includes a secret-like param which should be redacted in traces
    decision = {
        "intent": "test_run",
        "agents": [
            {"name": "Planner", "params": {}},
            {"name": "SQLGenerator", "params": {"sample_only": True, "api_key": "sk-TESTSECRET"}},
        ],
    }

    ctx = {"prompt": "test", "conn": None}
    out = run_decision_graph(decision, ctx)

    # We should get a node_traces list on return
    assert "node_traces" in out
    traces = out["node_traces"]
    assert isinstance(traces, list)
    # Find any trace that contains the API key param in input/meta and ensure it's redacted
    found = False
    for t in traces:
        # look at meta and input
        meta = t.get("meta", {})
        inp = t.get("input", {})
        if "api_key" in meta or (isinstance(inp, dict) and "params" in inp.get("params", {})):
            found = True
            # Ensure redaction placeholder present
            assert "<REDACTED>" in repr(meta.values()) or "<REDACTED>" in repr(inp.values())
    assert found, "expected to find a trace containing the secret-like param"


def test_redact_helper_behaviour():
    payload = {"a": "normal", "api_key": "sk-FOO12345", "nested": {"password": "hunter2"}}
    red = _redact(payload)
    assert red["a"] == "normal"
    assert red["api_key"] == "<REDACTED>"
    assert red["nested"]["password"] == "<REDACTED>"
