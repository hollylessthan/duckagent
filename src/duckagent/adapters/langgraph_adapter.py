"""LangGraph adapter shim for DuckAgent.

This module provides a thin adapter layer to run a decision graph using LangGraph
when available, and falls back to the local orchestrator for PoC and tests.

Usage:
  from duckagent.adapters.langgraph_adapter import run_decision_graph
  result = run_decision_graph(decision, context)

The adapter is intentionally minimal: it does not require LangGraph to be
installed for unit tests or local development. When LangGraph is installed,
the adapter will attempt to convert the decision into a LangGraph flow and
execute it (best-effort).
"""
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import LangGraph runtime or SDK. If not present, we run locally.
HAS_LANGGRAPH = False
langgraph = None
langgraph_sdk = None
try:
    import langgraph  # type: ignore
    langgraph = langgraph
    HAS_LANGGRAPH = True
except Exception:
    langgraph = None

try:
    import langgraph_sdk  # type: ignore
    langgraph_sdk = langgraph_sdk
    HAS_LANGGRAPH = True
except Exception:
    langgraph_sdk = None

# Local orchestrator fallback
from duckagent.orchestrator import execute as local_execute

# helper mapping
from duckagent.adapters.langgraph_mapping import build_run_payload


def run_decision_graph(decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Run a decision graph using LangGraph if available, otherwise fall back.

    decision: the Planner/Router decision dict containing `agents` list.
    context: runtime context, e.g. {'conn': conn, 'prompt': prompt}

    Returns the execution result dict.
    """
    if HAS_LANGGRAPH:
        # Prefer the SDK client path when available
        if langgraph_sdk is not None:
            try:
                run_payload = build_run_payload(decision)
                client = None
                # try common sdk helpers
                if hasattr(langgraph_sdk, "get_client"):
                    client = langgraph_sdk.get_client()
                else:
                    client_mod = getattr(langgraph_sdk, "client", None)
                    if client_mod and hasattr(client_mod, "LangGraphClient"):
                        client = client_mod.LangGraphClient()

                if client is not None:
                    # try common client run creation paths
                    if hasattr(client, "runs") and hasattr(client.runs, "create"):
                        created = client.runs.create(run_payload)
                        return {"langgraph_result": created}
                    if hasattr(client, "create_run"):
                        created = client.create_run(run_payload)
                        return {"langgraph_result": created}
                # if SDK client doesn't expose expected API, fall through
            except Exception:
                logger.exception("langgraph_sdk execution failed, will try runtime shim")

        # Fallback: if a runtime `langgraph` exposes a Graph API, try that
        if langgraph is not None and hasattr(langgraph, "Graph"):
            try:
                lg = langgraph
                graph = lg.Graph()
                nodes = {}
                for idx, node in enumerate(decision.get("agents", [])):
                    name = node.get("name")

                    def make_callable(n):
                        def fn(state, params=n.get("params", {})):
                            one_decision = {"agents": [n]}
                            return local_execute(one_decision, state)

                        return fn

                    node_id = f"node_{idx}_{name}"
                    nodes[node_id] = graph.add_node(name=node_id, fn=make_callable(node))

                node_ids = list(nodes.keys())
                for a, b in zip(node_ids, node_ids[1:]):
                    graph.add_edge(nodes[a], nodes[b])

                run_result = graph.run(context)
                return {"langgraph_result": run_result}
            except Exception:
                logger.exception("LangGraph runtime shim failed, falling back to local orchestrator")

        # final fallback to local orchestrator
        return local_execute(decision, context)
    else:
        logger.debug("LangGraph not available; using local orchestrator fallback")
        return local_execute(decision, context)


def build_graph_yaml(decision: Dict[str, Any]) -> str:
    """Return a tiny illustrative YAML representation of the decision graph.

    This is not a full LangGraph YAML but a readable representation the user
    can use as a starting point for a real LangGraph spec.
    """
    lines = ["graph:"]
    for i, node in enumerate(decision.get("agents", [])):
        name = node.get("name")
        params = node.get("params", {})
        lines.append(f"  - id: node_{i}")
        lines.append(f"    name: {name}")
        if params:
            lines.append(f"    params: {params}")
    return "\n".join(lines)
