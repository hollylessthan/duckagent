"""Example showing how to register a DataFrame in DuckDB via `Agent` and a custom table name.

Usage:
  python examples/run_with_dataframe_with_conn.py
"""
import pandas as pd
import duckdb
from duckagent.agent import Agent


def main():
    df = pd.DataFrame({"country": ["US", "CA", "US"], "revenue": [100, 200, 150]})

    # Create a DuckDB in-memory connection
    conn = duckdb.connect(':memory:')

    # Create Agent that will register the DataFrame under `sales_table`
    agent = Agent(conn=conn, table_name="sales_table")
    prompt = "Summarize revenue by country and show totals"
    res = agent.run(prompt, data=df)

    print("Decision:")
    print(res["decision"])
    print("\nSummary result:")
    print(res["execution"]["results"].get("Summarizer"))

    # You can also query the registered table directly
    print("\nRegistered table preview:")
    print(conn.execute("SELECT * FROM sales_table LIMIT 10").fetchdf())


if __name__ == '__main__':
    main()
