"""Example showing how to pass a pandas DataFrame into `Agent.run(..., data=...)`.

Usage:
  python examples/run_with_dataframe.py
"""
import pandas as pd
from duckagent.agent import Agent


def main():
    # Create a tiny DataFrame
    df = pd.DataFrame({"country": ["US", "CA", "US"], "revenue": [100, 200, 150]})

    # No DuckDB connection required for summarization path; Agent will prefer the
    # provided DataFrame and not attempt SQL generation.
    agent = Agent(conn=None)
    prompt = "Summarize revenue by country and highlight any anomalies."
    res = agent.run(prompt, data=df)

    print("Decision:")
    print(res["decision"])
    print("\nSummary result:")
    print(res["execution"]["results"].get("Summarizer"))


if __name__ == '__main__':
    main()
