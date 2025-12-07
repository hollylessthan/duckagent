from duckagent.adapters.langgraph_mapping import build_run_payload


def test_linear_mapping_defaults():
    decision = {"agents": [{"name": "A"}, {"name": "B"}, {"name": "C"}]}
    payload = build_run_payload(decision)
    items = payload["items"]
    assert len(items) == 3
    assert items[0]["id"] == "node_0"
    assert items[1]["depends_on"] == ["node_0"]
    assert items[1]["inputs"] == [{"from": "node_0", "output": "result"}]


def test_explicit_dependencies_and_ids():
    decision = {
        "agents": [
            {"id": "g1", "name": "Gen"},
            {"id": "r1", "name": "Run", "depends_on": ["g1"]},
        ]
    }
    payload = build_run_payload(decision)
    items = {i["id"]: i for i in payload["items"]}
    assert "g1" in items and "r1" in items
    assert items["r1"]["depends_on"] == ["g1"]
    assert items["r1"]["inputs"] == [{"from": "g1", "output": "result"}]


def test_custom_outputs_preserved_and_used():
    decision = {
        "agents": [
            {"id": "sql", "name": "SQLGen", "outputs": ["sql_text"]},
            {"id": "run", "name": "SQLRun", "depends_on": ["sql"]},
        ]
    }
    payload = build_run_payload(decision)
    items = {i["id"]: i for i in payload["items"]}
    assert items["sql"]["outputs"] == ["sql_text"]
    assert items["run"]["inputs"] == [{"from": "sql", "output": "sql_text"}]
