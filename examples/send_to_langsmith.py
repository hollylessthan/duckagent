# Example: send a graph produced from an internal decision to LangSmith
from duckagent.adapters.langgraph_adapter import LangGraphAdapter, LangGraphAdapterError
from duckagent.orchestrator import execute as local_execute

adapter = LangGraphAdapter()  # you can pass a real LangSmith client if you prefer

# If an OpenAI API key is provided via `OPENAI_API_KEY`, wire a tiny LLM
# wrapper into the orchestrator state so agents can call an LLM.
class SimpleOpenAI:
    def __init__(self, api_key: str):
        try:
            import openai

            openai.api_key = api_key
            self._openai = openai
        except Exception:
            raise RuntimeError("openai package not available; pip install openai to use LLM features")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        # Try ChatCompletions first, then Completion fallback
        try:
            resp = self._openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
            return resp.choices[0].message["content"]
        except Exception:
            try:
                resp = self._openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=max_tokens)
                return resp.choices[0].text
            except Exception as e:
                raise RuntimeError(f"OpenAI call failed: {e}")

decision = {
    "intent": "summarize_revenue",
    "agents": [
        {"name": "Planner", "params": {"prompt": "Summarize revenue by country"}},
        {"name": "SQLGenerator", "params": {"sample_only": True}},
        {"name": "SQLRunner", "params": {"engine": "duckdb", "table_name": "full_df"}},
        {"name": "Summarizer", "params": {"model": "small"}}
    ],
    "metadata": {"prompt": "Summarize revenue by country"}
}

# Convert to a LangGraph-like graph
graph = adapter.decision_to_graph(decision)

# Wire optional OpenAI LLM into context if `OPENAI_API_KEY` is set in env.
import os
openai_key = os.getenv("OPENAI_API_KEY")
context = {"conn": None, "prompt": decision.get("metadata", {}).get("prompt")}
if openai_key:
    try:
        llm = SimpleOpenAI(openai_key)
        context["llm"] = llm
    except Exception as e:
        print("WARNING: enable LLM failed:", e)

# Execute the decision locally to produce per-step outputs (so LangSmith can
# visualize actual run events/outputs rather than just the static graph).
exec_result = local_execute(decision, context)

# enrich the graph steps with outputs from the local run
enriched_graph = dict(graph)
nodes = enriched_graph.get("nodes", [])
results = exec_result.get("results", {})
for node in nodes:
    node_name = node.get("name")
    node["outputs"] = results.get(node_name)
enriched_graph["execution"] = exec_result

# Send enriched graph to LangSmith (returns a small dict like {'run_id': ..., 'run_url': ...})
try:
    # set debug=True to return the raw SDK response for troubleshooting
    resp = adapter.send_to_langsmith(enriched_graph, project="duckagent", debug=True)
    run_id = resp.get("run_id")
    run_url = resp.get("run_url")
    print("LangSmith run id:", run_id)
    if run_url:
        print("LangSmith run URL:", run_url)
    # if debug=True the adapter also returns a `raw` entry with the SDK response
    if resp.get("raw") is not None:
        print("LangSmith raw response:")
        print(resp.get("raw"))
except LangGraphAdapterError as e:
    print("LangSmith upload error:", e)