"""Decision -> LangGraph mapping helpers.

This module provides a minimal, testable transformation from the internal
`decision` dict (list of agents with optional params and dependencies) into a
LangGraph-compatible run payload. The mapping is intentionally conservative:
- Items are assigned stable ids `node_{i}`
- If an agent provides `depends_on` (list of ids) it is preserved; otherwise
  we wire nodes linearly by default.

The payload shape produced is a small dict suitable for use with
`langgraph_sdk` client APIs in the adapter. It is not a full LangGraph spec
but enough to run simple flows.
"""
from typing import Dict, Any, List


def build_run_payload(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Build a minimal run payload from a decision dict.

    decision: { 'agents': [ { 'name': 'SQLGenerator', 'params': {...}, 'id': 'gen1', 'depends_on': ['n0'] }, ... ] }

    Returns:
      { 'name': 'duckagent_decision', 'items': [ { 'id': 'node_0', 'name': 'SQLGenerator', 'params': {...}, 'depends_on': [] }, ... ] }
    """
    agents = decision.get("agents", []) or []
    items: List[Dict[str, Any]] = []

    # normalize ids and collect declared outputs
    id_to_outputs: Dict[str, List[str]] = {}
    for idx, a in enumerate(agents):
      aid = a.get("id") or f"node_{idx}"
      outputs = a.get("outputs") or a.get("provides") or ["result"]
      # ensure outputs is a list
      if isinstance(outputs, str):
        outputs = [outputs]
      id_to_outputs[aid] = list(outputs)
      items.append({
        "id": aid,
        "name": a.get("name", f"agent_{idx}"),
        "params": a.get("params", {}),
        # declare outputs explicitly for downstream mapping
        "outputs": list(outputs),
        # populate depends_on if provided, else empty for now
        "depends_on": list(a.get("depends_on", [])) if a.get("depends_on") is not None else [],
        # inputs will be filled below as references to upstream outputs
        "inputs": [],
      })

    # If no explicit dependencies are present, wire linearly
    if not any(item.get("depends_on") for item in items):
      for i in range(1, len(items)):
        items[i]["depends_on"] = [items[i - 1]["id"]]

    # Build inputs references for each item based on depends_on and upstream outputs
    for item in items:
      deps = item.get("depends_on") or []
      inputs = []
      for dep in deps:
        upstream_outputs = id_to_outputs.get(dep, ["result"])
        # default to first output as the primary connector
        inputs.append({"from": dep, "output": upstream_outputs[0]})
      item["inputs"] = inputs

    return {"name": "duckagent_decision", "items": items}
