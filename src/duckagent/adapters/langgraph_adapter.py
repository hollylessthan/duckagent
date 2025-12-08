"""LangGraph adapter shim for DuckAgent.

This module attempts to materialize Planner decisions into a LangGraph
runtime graph (when available) and otherwise falls back to the local
`orchestrator.execute`. It also collects per-node traces and can upload
redacted runs to LangSmith via `send_to_langsmith`.

The implementation is intentionally conservative: soft-imports optional
dependencies and never requires LangGraph/LangSmith for tests.
"""
from typing import Dict, Any, Callable, List, Optional
import logging
import os
import time

logger = logging.getLogger(__name__)

# Detect available runtimes/SDKs
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

# Mapping helper
from duckagent.adapters.langgraph_mapping import build_run_payload


class LangGraphAdapterError(Exception):
    """Adapter-level errors."""


class LangGraphUnavailable(LangGraphAdapterError):
    """Raised when a LangGraph runtime/client is not available for execution."""


class LangGraphAdapter:
    """Thin compatibility wrapper exposing the class-based API used in tests.

    The class delegates to the module-level functions implemented here.
    """

    def __init__(self, client: Any = None):
        self.client = client

    def decision_to_graph(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        return build_runtime_graph(decision, {})["graph_dict"]

    def graph_to_execution_result(self, exec_obj: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize different runtime shapes into the expected execution dict
        # Expected shape: {"execution": {"status": ..., "results": {...}, "metrics": {...}}, "trace_id": ...}
        status = exec_obj.get("status")
        results = exec_obj.get("nodes") or exec_obj.get("results") or {}
        metrics = exec_obj.get("metrics", {})
        trace_id = exec_obj.get("trace_id")
        return {"execution": {"status": status, "results": results, "metrics": metrics}, "trace_id": trace_id}

    def execute_graph(self, graph_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client and not HAS_LANGGRAPH:
            raise LangGraphUnavailable("No LangGraph client/runtime available")
        # If a client-like object was provided, try to call its create API
        if self.client is not None:
            try:
                if hasattr(self.client, "runs") and hasattr(self.client.runs, "create"):
                    return {"langgraph_result": self.client.runs.create(graph_payload)}
                if hasattr(self.client, "create_run"):
                    return {"langgraph_result": self.client.create_run(graph_payload)}
            except Exception:
                logger.exception("Provided LangGraph client failed; falling back to module runner")

        # Otherwise use module runner which will attempt runtime/SDK/fallback
        return run_decision_graph(graph_payload, {})

    def send_to_langsmith(self, graph: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return send_to_langsmith(graph, **kwargs)


def _redact_value(v: Any) -> Any:
    """Redact obvious secret-like strings, leave other values intact."""
    import re

    if isinstance(v, str):
        if re.search(r"sk-[A-Za-z0-9_\-]{8,}", v):
            return "<REDACTED>"
        return v
    return v


def _redact(obj: Any) -> Any:
    """Conservative recursive redaction for dict/list structures."""
    if isinstance(obj, dict):
        out = {}
        for k, val in obj.items():
            kl = k.lower()
            if any(s in kl for s in ("key", "secret", "token", "password", "api")):
                out[k] = "<REDACTED>"
                continue
            out[k] = _redact(val)
        return out
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return _redact_value(obj)


def build_runtime_graph(decision: Dict[str, Any], context: Dict[str, Any], agent_impls: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
    """Materialize a Planner decision into a runtime-friendly structure.

    Returns a dict with keys:
      - graph_dict: a serializable graph representation (nodes/edges/metadata)
      - runtime_nodes: list of (node_id, call_fn) for wiring into a runtime
      - edges: list of (from_id, to_id)

    `agent_impls` may provide concrete callables for agent names; otherwise
    the materializer will call the local orchestrator for each agent.
    """
    agents = decision.get("agents", []) or []
    nodes: List[Dict[str, Any]] = []
    runtime_nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []

    for i, a in enumerate(agents):
        name = a.get("name")
        params = a.get("params", {}) or {}
        node_id = f"node_{i}_{name}"
        nodes.append({"id": node_id, "name": name, "meta": params})

        # Determine callable: prefer agent_impls[name], else local_execute wrapper
        if agent_impls and name in agent_impls and callable(agent_impls[name]):
            call_fn = agent_impls[name]
        else:
            def make_local_fn(n):
                def fn(state, params=n.get("params", {})):
                    one_decision = {"agents": [n]}
                    return local_execute(one_decision, state)

                return fn

            call_fn = make_local_fn(a)

        runtime_nodes.append({"id": node_id, "name": name, "fn": call_fn, "meta": params})

    for a, b in zip(nodes, nodes[1:]):
        edges.append({"from": a["id"], "to": b["id"]})

    graph_dict = {"nodes": nodes, "edges": edges, "metadata": {"decision": {"intent": decision.get("intent")}}}
    return {"graph_dict": graph_dict, "runtime_nodes": runtime_nodes, "edges": edges}


def run_decision_graph(decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to run a decision via LangGraph runtime/SDK, falling back to local orchestrator.

    This function materializes the decision, attempts SDK/runtime execution,
    collects per-node traces (with redaction), and optionally uploads the
    redacted graph to LangSmith via `send_to_langsmith`.
    """
    materialized = build_runtime_graph(decision, context)
    graph_dict = materialized["graph_dict"]
    runtime_nodes = materialized["runtime_nodes"]

    traces: List[Dict[str, Any]] = []

    # SDK path: if langgraph_sdk available, try to build a payload and create run
    if langgraph_sdk is not None:
        try:
            payload = build_run_payload(decision)
            client = None
            if hasattr(langgraph_sdk, "get_client"):
                client = langgraph_sdk.get_client()
            else:
                client_mod = getattr(langgraph_sdk, "client", None)
                if client_mod and hasattr(client_mod, "LangGraphClient"):
                    client = client_mod.LangGraphClient()

            if client is not None:
                if hasattr(client, "runs") and hasattr(client.runs, "create"):
                    created = client.runs.create(payload)
                    # best-effort: attach traces if possible
                    return {"langgraph_result": created}
                if hasattr(client, "create_run"):
                    created = client.create_run(payload)
                    return {"langgraph_result": created}
        except Exception:
            logger.exception("langgraph_sdk execution failed; will try runtime shim")

    # Runtime shim path: if runtime is installed, wire callables and collect traces
    if HAS_LANGGRAPH and langgraph is not None and hasattr(langgraph, "Graph"):
        try:
            lg = langgraph
            g = lg.Graph()

            # closure to collect traces for each node
            def make_traced_fn(node_meta):
                fn = node_meta["fn"]

                def wrapped(state):
                    started = time.time()
                    in_payload = {"state": state, "params": node_meta.get("meta", {})}
                    try:
                        out = fn(state)
                        status = "success"
                    except Exception as e:
                        out = {"error": str(e)}
                        status = "error"
                    ended = time.time()
                    trace = {
                        "id": node_meta["id"],
                        "name": node_meta.get("name"),
                        "meta": node_meta.get("meta", {}),
                        "input": _redact(in_payload),
                        "output": _redact(out),
                        "status": status,
                        "started_at": started,
                        "ended_at": ended,
                        "duration": ended - started,
                    }
                    traces.append(trace)
                    if status == "error":
                        raise Exception(out.get("error"))
                    return out

                return wrapped

            node_objs = {}
            for n in runtime_nodes:
                node_id = n["id"]
                node_objs[node_id] = g.add_node(name=node_id, fn=make_traced_fn(n))

            # wire edges linearly as materializer produced them
            for e in graph_dict.get("edges", []):
                a = e.get("from")
                b = e.get("to")
                if a in node_objs and b in node_objs:
                    g.add_edge(node_objs[a], node_objs[b])

            run_result = g.run(context)

            # attach traces into graph dict for optional upload
            upload_graph = dict(graph_dict)
            upload_graph["node_traces"] = traces
            try:
                # best-effort upload if env present
                if os.getenv("LANGSMITH_API_KEY"):
                    send_to_langsmith(upload_graph, debug=False)
            except Exception:
                logger.exception("send_to_langsmith failed (non-fatal)")

            return {"langgraph_result": run_result, "node_traces": traces}
        except Exception:
            logger.exception("LangGraph runtime shim failed; falling back to local orchestrator")

    # No runtime available or runtime failed: execute locally node-by-node
    logger.debug("Running decision locally via orchestrator with per-node tracing")
    local_state = dict(context or {})
    local_results = {}
    for n in runtime_nodes:
        node_id = n["id"]
        name = n.get("name")
        fn = n["fn"]
        started = time.time()
        in_payload = {"state": local_state, "params": n.get("meta", {})}
        try:
            out = fn(local_state)
            status = "success"
        except Exception as e:
            out = {"error": str(e)}
            status = "error"
        ended = time.time()
        trace = {
            "id": node_id,
            "name": name,
            "meta": n.get("meta", {}),
            "input": _redact(in_payload),
            "output": _redact(out),
            "status": status,
            "started_at": started,
            "ended_at": ended,
            "duration": ended - started,
        }
        traces.append(trace)
        local_results[node_id] = {"status": status, "output": out}
        if status == "error":
            # stop on node error (same behavior as linear graph)
            break

    # Build an uploadable graph representation with traces
    upload_graph = dict(graph_dict)
    upload_graph["node_traces"] = traces
    try:
        if os.getenv("LANGSMITH_API_KEY"):
            send_to_langsmith(upload_graph, debug=False)
    except Exception:
        logger.exception("send_to_langsmith failed (non-fatal)")

    return {"execution": {"status": "completed", "results": local_results}, "node_traces": traces}


def send_to_langsmith(
    graph: Dict[str, Any],
    project: str = "duckagent",
    api_key_env: str = "LANGSMITH_API_KEY",
    debug: bool = False,
) -> Dict[str, Any]:
    """Send a graph to LangSmith for visualization.

    Soft-imports the `langsmith` SDK and tries several client shapes.
    Returns {'run_id': id, 'run_url': url} on success.
    """
    try:
        import langsmith  # type: ignore
    except Exception:
        raise LangGraphAdapterError("LangSmith SDK (langsmith) is not installed")

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise LangGraphAdapterError(f"Missing {api_key_env} environment variable")

    sanitized = _redact(graph)

    Client = getattr(langsmith, "Client", None)
    client = None
    if Client is not None:
        client = Client(api_key=api_key)

    decision_meta = graph.get("metadata", {}).get("decision", {})
    name = decision_meta.get("intent") or "duckagent_run"
    steps = []
    for node in graph.get("nodes", []):
        steps.append({"id": node.get("id"), "name": node.get("name"), "meta": node.get("meta", {})})

    run_payload = {"name": name, "project": project, "steps": steps, "graph": sanitized, "run_type": "chain"}

    # client.runs.create
    if client is not None and hasattr(client, "runs") and hasattr(client.runs, "create"):
        try:
            run = client.runs.create(run_payload)
            run_id = getattr(run, "id", None) or run_payload.get("name")
            run_url = getattr(run, "url", None) if hasattr(run, "url") else None
            if run_url is None and isinstance(run, dict):
                run_url = run.get("url") or run.get("view_url") or run.get("browser_url")
            if run_url is None and run_id:
                run_url = f"https://smith.langchain.com/projects/{project}/runs/{run_id}"
            res = {"run_id": run_id, "run_url": run_url}
            if debug:
                try:
                    res["raw"] = run.to_dict() if hasattr(run, "to_dict") else repr(run)
                except Exception:
                    res["raw"] = repr(run)
            return res
        except Exception as e:
            logger.exception("client.runs.create failed")
            raise LangGraphAdapterError(f"langsmith client.runs.create failed: {e}")

    # client.create_run
    if client is not None and hasattr(client, "create_run"):
        try:
            run = client.create_run(inputs=run_payload, run_type="chain", name=name, project=project)
            run_id = getattr(run, "id", None) or run_payload.get("name")
            run_url = getattr(run, "url", None) if hasattr(run, "url") else None
            if run_url is None and isinstance(run, dict):
                run_url = run.get("url") or run.get("view_url") or run.get("browser_url")
            if run_url is None and run_id:
                run_url = f"https://smith.langchain.com/projects/{project}/runs/{run_id}"
            res = {"run_id": run_id, "run_url": run_url}
            if debug:
                try:
                    res["raw"] = run.to_dict() if hasattr(run, "to_dict") else repr(run)
                except Exception:
                    res["raw"] = repr(run)
            return res
        except Exception as e:
            logger.exception("client.create_run failed")
            raise LangGraphAdapterError(f"langsmith client.create_run failed: {e}")

    # module-level create
    if hasattr(langsmith, "create"):
        try:
            run = langsmith.create(run_payload)
            run_id = getattr(run, "id", None) or run_payload.get("name")
            run_url = getattr(run, "url", None) if hasattr(run, "url") else None
            if run_url is None and isinstance(run, dict):
                run_url = run.get("url") or run.get("view_url") or run.get("browser_url")
            if run_url is None and run_id:
                run_url = f"https://smith.langchain.com/projects/{project}/runs/{run_id}"
            res = {"run_id": run_id, "run_url": run_url}
            if debug:
                try:
                    res["raw"] = run.to_dict() if hasattr(run, "to_dict") else repr(run)
                except Exception:
                    res["raw"] = repr(run)
            return res
        except Exception as e:
            logger.exception("langsmith.create failed")
            raise LangGraphAdapterError(f"langsmith.create failed: {e}")

    raise LangGraphAdapterError("LangSmith client does not provide a supported create API")


def build_graph_yaml(decision: Dict[str, Any]) -> str:
    """Return a tiny illustrative YAML representation of the decision graph."""
    lines = ["graph:"]
    for i, node in enumerate(decision.get("agents", [])):
        name = node.get("name")
        params = node.get("params", {})
        lines.append(f"  - id: node_{i}")
        lines.append(f"    name: {name}")
        if params:
            lines.append(f"    params: {params}")
    return "\n".join(lines)
"""LangGraph adapter shim for DuckAgent.

Provides a minimal adapter that will attempt to run a decision via an
installed LangGraph runtime or SDK and falls back to the local
orchestrator. Also includes a helper to upload a graph to LangSmith for
visualization/tracing (soft-imports the `langsmith` SDK).

The module is intentionally conservative about dependencies so unit tests
and local development do not require LangGraph or LangSmith to be
installed.
"""
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Detect available runtimes/SDKs
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
