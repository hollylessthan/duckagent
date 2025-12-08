"""Demo: run a Planner -> LangGraph adapter supervised execution using MockLLM.

This demo shows how to produce a decision via the `Planner` (using
`MockLLM`), and then execute the decision via `langgraph_adapter.run_decision_graph`.
This will exercise the adapter's runtime shim (which falls back to the local
orchestrator when LangGraph is not installed).

Run locally:
    python3 examples/langgraph_supervisor_demo.py

No external dependencies required — uses MockLLM and the local orchestrator.
"""
from duckagent.planner import Planner
from duckagent.llm_adapter import MockLLM
from duckagent.adapters import langgraph_adapter
from duckagent.agent import Agent


def main():
    # Use MockLLM to generate a deterministic JSON decision
    mock = MockLLM()

    # Prepare a mocked LLM.generate that returns a JSON decision for 'analyze'
    decision = {
        "intent": "analyze",
        "agents": [
            {"name": "SQLGenerator", "params": {"max_rows": 10}},
            {"name": "SQLRunner", "params": {}},
            {"name": "AnalysisAgent", "params": {}},
            {"name": "Summarizer", "params": {}},
        ],
        "hints": {"sample_only": True},
    }

    # Mock.generate should return a JSON string
    def gen(prompt, **opts):
        import json

        return json.dumps(decision)

    mock.generate = gen

    planner = Planner(llm=mock)

    # Context: no DB connection for this demo; orchestrator will produce stub previews
    ctx = {"prompt": "Analyze sales trends for the dataset.", "conn": None}

    # Get a validated decision from the planner
    dec = planner.plan_for_intent("analyze", "Analyze sales trends", ctx)
    print("Planner decision:")
    import json

    print(json.dumps(dec, indent=2))

    # Execute the decision via the LangGraph adapter. The adapter will:
    # - attempt to use langgraph SDK/runtime if present
    # - otherwise fall back to local orchestrator execution
    print("\nRunning decision via langgraph_adapter... (may run locally if LangGraph not installed)")
    out = langgraph_adapter.run_decision_graph(dec, ctx)

    print("\nExecution result:\n")
    print(out)


if __name__ == "__main__":
    main()
