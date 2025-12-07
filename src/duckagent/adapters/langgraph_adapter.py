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
    HAS_LANGGRAPH = True
except Exception:
    langgraph = None

try:
    import langgraph_sdk  # type: ignore
    # presence of langgraph_sdk is enough to consider LangGraph available
    langgraph_sdk = langgraph_sdk
    HAS_LANGGRAPH = True
except Exception:
    langgraph_sdk = None

# Local orchestrator fallback
from duckagent.orchestrator import execute as local_execute


def run_decision_graph(decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Run a decision graph using LangGraph if available, otherwise fall back.

    decision: the Planner/Router decision dict containing `agents` list.
    context: runtime context, e.g. {'conn': conn, 'prompt': prompt}

    Returns the execution result dict.
    """
    if HAS_LANGGRAPH:
        # Prefer an SDK-driven execution path when available. This tries several
        # common LangGraph SDK entrypoints (best-effort) and falls back to the
        # older dynamic-Graph approach or the local orchestrator if unavailable.
        # The adapter intentionally tolerates many failure modes and falls back.
        # 1) Try langgraph_sdk client-based run submission
        if langgraph_sdk is not None:
            try:
                try:
                    client = None
                    # prefer a helper if present
                    if hasattr(langgraph_sdk, "get_client"):
                        client = langgraph_sdk.get_client()
                    else:
                        # some versions expose client module with LangGraphClient
                        client_cls = getattr(langgraph_sdk, "client", None)
                        if client_cls and hasattr(client_cls, "LangGraphClient"):
                            client = client_cls.LangGraphClient()

                    # Build a minimal run payload from the decision. We construct
                    # a simple list of node items; richer mapping should be added
                    # in future iterations.
                    run_payload = {
                        "name": "duckagent_decision",
                        "items": [],
                    }
                    for idx, node in enumerate(decision.get("agents", [])):
                        run_payload["items"].append({"id": f"node_{idx}", "name": node.get("name"), "params": node.get("params", {})})

                    # Try a few common client interfaces
                    if client is not None:
                        # prefer `runs` collection
                        if hasattr(client, "runs") and hasattr(client.runs, "create"):
                            created = client.runs.create(run_payload)
                            return {"langgraph_result": created}
                        # fallback to a top-level create_run
                        if hasattr(client, "create_run"):
                            created = client.create_run(run_payload)
                            return {"langgraph_result": created}

                except Exception:
                    # fall through to next strategy
                    logger.debug("langgraph_sdk client path failed, trying legacy shim")
            except Exception:
                logger.exception("langgraph_sdk execution attempt failed; falling back")

        # 2) If a runtime `langgraph` package exposes a programmatic Graph API,
        #    try to use it as a best-effort. This mirrors the previous shim.
        try:
            if langgraph is not None and hasattr(langgraph, "Graph"):
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
        except Exception as e:
            logger.exception("LangGraph runtime shim failed, falling back to local orchestrator: %s", e)

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
