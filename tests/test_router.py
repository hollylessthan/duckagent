import pytest

from duckagent.router import Router


def test_router_detects_analysis_intent():
    r = Router()
    res = r.detect_intent("Please analyze monthly revenue and drivers")
    assert res["intent"] == "analyze"
    assert "AnalysisAgent" in res["agents"]


def test_router_detects_sql_intent():
    r = Router()
    res = r.detect_intent("Show top 10 products by sales")
    assert res["intent"] == "sql"
    assert "SQLGenerator" in res["agents"]


def test_router_detects_summarize_intent():
    r = Router()
    res = r.detect_intent("Give me a summary of the dataframe")
    assert res["intent"] == "summarize"
    assert "Summarizer" in res["agents"]


def test_router_sum_vs_summary():
    r = Router()
    res_sum = r.detect_intent("Calculate sum of sales for last month")
    assert res_sum["intent"] == "sql"
    res_summary = r.detect_intent("Provide a summary of the sales dataframe")
    assert res_summary["intent"] == "summarize"


def test_router_does_not_false_positive_on_assumption():
    r = Router()
    res = r.detect_intent("Check the assumption about the data distribution")
    # should not be classified as SQL or Summarize; fallback to unknown or planner flow
    assert res["intent"] not in ("sql", "summarize")
