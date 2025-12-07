"""Streamlit demo for DuckAgent PoC.

Run with:
    streamlit run examples/streamlit_demo.py

Features:
- Upload CSV
- Optionally register the DataFrame into an in-memory DuckDB connection under a configurable table name
- Run `Agent.run(..., data=...)` and display decision, summary and preview
"""
import io
import pandas as pd

try:
    import streamlit as st
except Exception:
    raise RuntimeError("Streamlit must be installed to run the demo. Install with `pip install streamlit`.")

from duckagent.agent import Agent


def main():
    st.set_page_config(page_title="DuckAgent Demo", layout="wide")
    st.title("DuckAgent — Streamlit Demo")

    st.markdown(
        "Upload a CSV and run the assistant. The demo will prefer summarizing the provided DataFrame."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    register_to_duckdb = st.checkbox("Register uploaded DataFrame into DuckDB (for SQL paths)", value=False)
    table_name = st.text_input("Table name (when registering)", value="full_df")

    llm_choice = st.selectbox("LLM adapter (PoC)", ["none", "mock"], index=1)

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            data = pd.read_csv(io.TextIOWrapper(uploaded, encoding="utf-8"))

        st.subheader("Data preview")
        st.dataframe(data.head(20))

        prompt = st.text_area("Prompt", value="Summarize the dataset and highlight anomalies.")

        if st.button("Run Agent"):
            # Create Agent
            conn = None
            if register_to_duckdb:
                import duckdb

                conn = duckdb.connect(":memory:")

            agent = Agent(conn=conn, table_name=table_name)

            # For PoC, allow using mock LLM if available on the adapter layer
            llm = None
            if llm_choice == "mock":
                try:
                    from duckagent.llm_adapter import MockLLM

                    llm = MockLLM()
                except Exception:
                    llm = None

            # Run agent
            res = agent.run(prompt, data=data, context={"llm": llm} if llm else None)

            st.subheader("Decision")
            st.json(res.get("decision"))

            st.subheader("Execution results")
            exec_results = res.get("execution", {}).get("results", {})
            st.write(exec_results)

            summary = res.get("execution", {}).get("summary_text") or exec_results.get("Summarizer", {}).get("summary")
            if summary:
                st.subheader("Summary")
                st.write(summary)

            # Show registered table if applicable
            if conn is not None:
                try:
                    st.subheader("Registered table preview")
                    tbl = conn.execute(f"SELECT * FROM {table_name} LIMIT 20").fetchdf()
                    st.dataframe(tbl)
                except Exception as e:
                    st.error(f"Failed to preview registered table: {e}")


if __name__ == "__main__":
    main()
