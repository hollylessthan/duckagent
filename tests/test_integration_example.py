import pandas as pd
import duckdb
from duckagent.agent import Agent


def test_run_with_dataframe_and_conn():
    # prepare a small DataFrame
    df = pd.DataFrame({"country": ["US", "CA", "US"], "revenue": [100, 200, 150]})

    # in-memory DuckDB connection
    conn = duckdb.connect(':memory:')

    # create Agent that will register the DataFrame under `sales_table`
    agent = Agent(conn=conn, table_name="sales_table")
    prompt = "Summarize revenue by country and show totals"
    res = agent.run(prompt, data=df)

    # ensure the table was registered and contains the same number of rows
    tbl = conn.execute("SELECT * FROM sales_table").fetchdf()
    assert len(tbl) == len(df)

    # ensure Summarizer produced a summary when summarization path is chosen
    summ = res["execution"]["results"].get("Summarizer")
    assert isinstance(summ, dict)
    assert "summary" in summ

    # sanity-check that the registered table can be queried (aggregate example)
    agg = conn.execute("SELECT country, SUM(revenue) as total FROM sales_table GROUP BY country ORDER BY country").fetchdf()
    assert "total" in agg.columns