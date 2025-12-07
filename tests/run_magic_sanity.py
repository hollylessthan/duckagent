import pandas as pd
import duckdb
from duckagent.agent import Agent

# Simple sanity script that mirrors the IPython magic's runtime path
# It creates a DuckDB connection, registers a DataFrame, constructs an Agent,
# runs a short prompt, and prints the result.

def main():
    df = pd.DataFrame({"country": ["US","CA"], "revenue": [100, 200]})

    conn = duckdb.connect(':memory:')
    # register DataFrame so SQL-based agents can reference it if needed
    try:
        conn.register('sales_df', df)
    except Exception:
        # some duckdb versions may not support register; ignore for sanity test
        pass

    agent = Agent(conn=conn, table_name='sales_df')
    res = agent.run('Summarize revenue by country', data=df)
    print('Agent.run output keys:', list(res.keys()))
    print('Execution (pretty):')
    import pprint
    pprint.pprint(res.get('execution'))


if __name__ == '__main__':
    main()
