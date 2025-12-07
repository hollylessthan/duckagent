# CHANGELOG

## Unreleased

- Add `data` parameter to `Agent.run(...)` allowing callers to pass a Pandas DataFrame directly.
- `Agent` accepts optional `table_name` to control DuckDB registration name for provided DataFrames (default `full_df`).
- `Planner.plan_for_intent` prefers a Summarizer-only plan when `context` contains `full_df` or `rows_preview`.
- `Summarizer` now prefers summarizing a provided DataFrame, uses LLM when available, and returns a helpful message when no data present.
- Add unit tests covering data handling (`tests/test_data_handling.py`).
- Add example `examples/run_with_dataframe.py` demonstrating `Agent.run(..., data=...)`.
 - Add `use_langgraph` opt-in flag to `Agent` to run decision graphs via LangGraph when available; falls back to local orchestrator.
 - Add unit tests for `use_langgraph` behavior and update README example.
 - feat(langgraph): add `langgraph` SDK support, `langgraph_mapping` (decision→run payload), adapter fallbacks, and mapping unit tests.

