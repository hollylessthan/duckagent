# Changelog

## Unreleased (2025-12-05)

- Add `data` parameter to `Agent.run(...)` to accept a Pandas DataFrame and inject it into runtime context as `full_df`.
- Planner now prefers a Summarizer-only plan when `context` contains `full_df` or a `rows_preview`.
- Summarizer updated to prefer summarizing a provided DataFrame, use LLM when available, and return a helpful fallback message when no data is present.
- Added unit tests: `test_agent_run_with_data_param`, `test_planner_detects_context_df`, `test_summarizer_no_data_message`.
- All tests passing locally (`8 passed`).

