"""Full supervisor-style demo with LangSmith tracing.

This script demonstrates a simple supervisor that coordinates two subagents
(`invoice` and `music`) by running their decisions sequentially, collecting
node-level traces, and uploading a redacted graph to LangSmith via the
`langgraph_adapter.send_to_langsmith` helper.

Requirements to use real OpenAI + LangSmith:
  pip install openai langsmith
  export OPENAI_API_KEY=... 
  export LANGSMITH_API_KEY=...

The script will fall back to `MockLLM` if OpenAI or keys are not available so
you can test the flow offline.
"""
import os
import json
from typing import Dict, Any, List

from duckagent.llm_adapter import MockLLM, OpenAIAdapter
from duckagent.planner import Planner
from duckagent.adapters import langgraph_adapter


def choose_llm():
    # Prefer OpenAIAdapter when API key and package available, else MockLLM
    key = os.getenv("OPENAI_API_KEY")
    if key:
        try:
            llm = OpenAIAdapter(api_key=key)
            print("Using OpenAIAdapter for LLM calls")
            return llm
        except Exception as e:
            print("OpenAIAdapter unavailable or import failed, falling back to MockLLM:", e)
    print("Using MockLLM for offline testing")
    return MockLLM()


def collect_local_traces(decision: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a minimal graph payload with node-level execution metadata.

    exec_result is expected to be the local orchestrator result (dict with
    'results' mapping agent name -> output). We map each agent -> node and
    attach outputs as node meta for LangSmith upload.
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    agents = decision.get("agents", []) or []
    for i, a in enumerate(agents):
        name = a.get("name")
        node_id = f"node_{i}_{name}"
        node_meta = {"params": a.get("params", {})}
        # attach execution result if present
        results = exec_result.get("results", {})
        node_exec = results.get(name)
        if node_exec is not None:
            node_meta["execution"] = node_exec
        nodes.append({"id": node_id, "name": name, "meta": node_meta})
        if i > 0:
            edges.append({"from": f"node_{i-1}_{agents[i-1].get('name')}", "to": node_id})

    payload = {"nodes": nodes, "edges": edges, "metadata": {"decision": {"intent": decision.get("intent")}}}
    return payload


def run_supervisor_flow():
    llm = choose_llm()
    planner = Planner(llm=llm)

    # For simplicity, the supervisor will run two subagent decisions sequentially.
    # Use domain-focused example decisions (data/SQL flows) rather than
    # unrelated placeholders.
    summary_decision = {
        "intent": "data_summary",
        "agents": [
            {"name": "Planner", "params": {}},
            {"name": "SQLGenerator", "params": {"sample_only": True}},
            {"name": "SQLRunner", "params": {"max_rows": 5}},
            {"name": "Summarizer", "params": {}},
        ],
        "hints": {},
    }

    inspect_decision = {
        "intent": "schema_inspect",
        "agents": [
            {"name": "Planner", "params": {}},
            {"name": "SQLGenerator", "params": {"sample_only": True}},
            {"name": "SQLRunner", "params": {"max_rows": 10}},
            {"name": "Summarizer", "params": {}},
        ],
        "hints": {},
    }

    ctx = {"prompt": "Supervisor orchestrating invoice and music subagents", "conn": None}

    # Execute invoice decision via adapter (uses runtime if available, else local)
    # Run the summary decision
    inv_out = langgraph_adapter.run_decision_graph(summary_decision, ctx)
    print("Data summary execution result:\n", inv_out)

    # Build trace payload for the data summary
    if "langgraph_result" in inv_out:
        invoice_payload = {"nodes": [], "edges": [], "metadata": {"decision": {"intent": "data_summary"}}}
    else:
        invoice_payload = collect_local_traces(summary_decision, inv_out)

    # Execute music decision
    # Run the schema inspection decision
    mus_out = langgraph_adapter.run_decision_graph(inspect_decision, ctx)
    print("Schema inspect execution result:\n", mus_out)

    if "langgraph_result" in mus_out:
        music_payload = {"nodes": [], "edges": [], "metadata": {"decision": {"intent": "schema_inspect"}}}
    else:
        music_payload = collect_local_traces(inspect_decision, mus_out)

    # Combine into a supervisor-level graph for visualization: concatenate nodes and wire
    combined_nodes = invoice_payload.get("nodes", []) + music_payload.get("nodes", [])
    combined_edges = invoice_payload.get("edges", [])
    # Connect last invoice node to first music node if both present
    if invoice_payload.get("nodes") and music_payload.get("nodes"):
        combined_edges += music_payload.get("edges", [])
        combined_edges.append({"from": invoice_payload["nodes"][-1]["id"], "to": music_payload["nodes"][0]["id"]})
    else:
        combined_edges += music_payload.get("edges", [])

    supervisor_graph = {"nodes": combined_nodes, "edges": combined_edges, "metadata": {"decision": {"intent": "supervisor_flow"}}}

    print("Prepared combined graph for LangSmith (redacted on send):")
    print(json.dumps(supervisor_graph, indent=2))

    # Attempt to send to LangSmith (soft-imported inside the helper)
    try:
        res = langgraph_adapter.send_to_langsmith(supervisor_graph, project=os.getenv("LANGSMITH_PROJECT", "duckagent"), debug=True)
        print("LangSmith upload result:", res)
    except Exception as e:
        print("LangSmith upload skipped/failed:", e)


if __name__ == "__main__":
    run_supervisor_flow()
