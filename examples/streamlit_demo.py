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

    llm_choice = st.selectbox("LLM adapter (PoC)", ["none", "mock", "openai"], index=1)

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            data = pd.read_csv(io.TextIOWrapper(uploaded, encoding="utf-8"))

        st.subheader("Data preview")
        st.dataframe(data.head(20))

        prompt = st.text_area("Prompt", value="Summarize the dataset and highlight anomalies.")
        allow_raw = st.checkbox("Allow LLM to see raw data (unsafe)", value=False)

        def _redact_value(v):
            import re

            if isinstance(v, str):
                # redact email-like strings and API-key like tokens
                if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", v):
                    return "<REDACTED_EMAIL>"
                if re.search(r"sk-[A-Za-z0-9_\-]{8,}", v):
                    return "<REDACTED_KEY>"
                return v
            # numeric values left as-is for simple analysis
            return v

        def build_safe_summary(df, nrows=5):
            cols = list(df.columns)
            n_rows, n_cols = df.shape
            sample = df.head(nrows).copy()
            for c in sample.columns:
                sample[c] = sample[c].apply(_redact_value)
            return {
                "n_rows": int(n_rows),
                "n_cols": int(n_cols),
                "columns": cols,
                "sample_rows": sample.to_dict(orient="records"),
            }

        if st.button("Run Agent"):
            # Create Agent
            conn = None
            if register_to_duckdb:
                import duckdb

                conn = duckdb.connect(":memory:")

            agent = Agent(conn=conn, table_name=table_name)

            # For PoC, allow using mock LLM if available on the adapter layer
            # or use OpenAI if `OPENAI_API_KEY` is present and selected.
            llm = None
            if llm_choice == "mock":
                try:
                    from duckagent.llm_adapter import MockLLM

                    llm = MockLLM()
                except Exception:
                    llm = None
            elif llm_choice == "openai":
                # try to wire a very small OpenAI wrapper if the package
                # is installed and the env var is set
                import os

                openai_key = os.getenv("OPENAI_API_KEY")
                if openai_key:
                    try:
                        # simple wrapper inline to avoid new module dependency
                        class _SimpleOpenAI:
                            def __init__(self, api_key: str):
                                import openai

                                openai.api_key = api_key
                                self._openai = openai

                            def generate(self, prompt: str, max_tokens: int = 256) -> str:
                                try:
                                    resp = self._openai.ChatCompletion.create(
                                        model="gpt-3.5-turbo",
                                        messages=[{"role": "user", "content": prompt}],
                                        max_tokens=max_tokens,
                                    )
                                    return resp.choices[0].message["content"]
                                except Exception:
                                    resp = self._openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=max_tokens)
                                    return resp.choices[0].text

                        llm = _SimpleOpenAI(openai_key)
                    except Exception as e:
                        st.warning(f"Failed to initialize OpenAI LLM: {e}")

            # Decide whether to send raw data to the LLM. If the user opts out
            # we build a small redacted summary and include that in the prompt
            # instead of passing the full DataFrame to the agent/LLM.
            if allow_raw:
                summary_note = "LLM allowed to see raw data."
                run_data = data
                run_prompt = prompt
            else:
                summary_note = "LLM will NOT see raw data; sending redacted summary instead."
                safe_summary = build_safe_summary(data)
                run_data = None
                # embed the safe summary into the prompt so downstream agents
                # can reason about the dataset without seeing raw values
                run_prompt = f"{prompt}\n\nData summary (redacted): {safe_summary}"

            st.info(summary_note)

            # Show what will be sent to the LLM (sanitized) for transparency
            st.subheader("What will be sent to the LLM")
            if allow_raw:
                st.write("Raw DataFrame (first 5 rows):")
                st.dataframe(data.head(5))
            else:
                st.write(safe_summary)

            # Run agent
            res = agent.run(run_prompt, data=run_data, context={"llm": llm} if llm else None)

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
