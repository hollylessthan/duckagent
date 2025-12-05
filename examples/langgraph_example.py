"""Example showing how to run a decision graph via the langgraph adapter.

Usage:
  python examples/langgraph_example.py

This example uses the `Agent` facade to produce a decision, then runs the
decision graph with the adapter. If LangGraph is not installed, the adapter
falls back to the local orchestrator.
"""
from duckagent.agent import Agent
from duckagent.adapters.langgraph_adapter import run_decision_graph, build_graph_yaml
import duckdb


def main():
    # create a tiny in-memory DuckDB and example table
    conn = duckdb.connect(':memory:')
    conn.execute("CREATE TABLE sample_table (id INTEGER, val DOUBLE, country VARCHAR)")
    conn.execute("INSERT INTO sample_table VALUES (1, 10.0, 'US'), (2, 20.0, 'CA')")

    agent = Agent(conn=conn)
    prompt = "Analyze monthly revenue trend by country, highlight anomalies over the past 12 months"
    decision = agent.router.detect_intent(prompt)
    # get a richer plan from planner
    decision = agent.planner.plan_for_intent(decision.get('intent'), prompt, {'conn': conn})

    print("Decision graph (YAML-like):")
    print(build_graph_yaml(decision))

    print('\nRunning decision graph via LangGraph adapter (or local fallback)...')
    result = run_decision_graph(decision, {'conn': conn, 'prompt': prompt})
    print('\nResult:')
    print(result)


if __name__ == '__main__':
    main()
