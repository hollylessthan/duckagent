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

# Try to import LangGraph runtime. If not present, we run locally.
try:
    import langgraph  # type: ignore
    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False

# Local orchestrator fallback
from duckagent.orchestrator import execute as local_execute


def run_decision_graph(decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Run a decision graph using LangGraph if available, otherwise fall back.

    decision: the Planner/Router decision dict containing `agents` list.
    context: runtime context, e.g. {'conn': conn, 'prompt': prompt}

    Returns the execution result dict.
    """
    if HAS_LANGGRAPH:
        try:
            # Best-effort mapping of decision -> LangGraph flow
            # We avoid depending on specific LangGraph APIs here; try a common pattern.
            lg = langgraph
            # Create a graph and nodes dynamically
            graph = lg.Graph()
            nodes = {}
            for idx, node in enumerate(decision.get("agents", [])):
                name = node.get("name")
                # wrap a small function that will call our local implementations
                def make_callable(n):
                    def fn(state, params=n.get("params", {})):
                        # state is expected to be the shared context dict
                        # call local orchestrator per-node handlers via our AGENT_IMPL mapping
                        # To keep adapter simple, run the node using local orchestrator helpers
                        # by creating a tiny one-node decision and executing it.
                        one_decision = {"agents": [n]}
                        return local_execute(one_decision, state)

                    return fn

                node_id = f"node_{idx}_{name}"
                nodes[node_id] = graph.add_node(name=node_id, fn=make_callable(node))

            # wire nodes in a linear fashion
            node_ids = list(nodes.keys())
            for a, b in zip(node_ids, node_ids[1:]):
                graph.add_edge(nodes[a], nodes[b])

            # execute the graph
            run_result = graph.run(context)
            return {"langgraph_result": run_result}
        except Exception as e:
            logger.exception("LangGraph execution failed, falling back to local orchestrator: %s", e)
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
