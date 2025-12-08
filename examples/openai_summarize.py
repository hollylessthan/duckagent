"""Example: Run the Summarizer using the OpenAIAdapter.

Requirements:
- `pip install openai pandas`
- `export OPENAI_API_KEY=...` (or set in your environment)

Run:
    python3 examples/openai_summarize.py

This script constructs a small sample DataFrame and passes it into
`Agent.run(..., data=...)` with an `OpenAIAdapter` instance so the
summarizer will call the real OpenAI API.
"""
from duckagent.agent import Agent
from duckagent.llm_adapter import OpenAIAdapter
import pandas as pd
import os
import sys

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: Please set OPENAI_API_KEY in your environment.")
    sys.exit(1)

# Initialize the OpenAIAdapter with the API key. You can also omit the
# api_key argument if you already set the env var.
llm = OpenAIAdapter(api_key=API_KEY, model="gpt-4o-mini")

# Create a short sample DataFrame similar to the `tips` dataset used earlier
sample = pd.DataFrame([
    {"total_bill": 16.99, "tip": 1.01, "sex": "Female", "smoker": "No", "day": "Sun", "time": "Dinner", "size": 2},
    {"total_bill": 10.34, "tip": 1.66, "sex": "Male", "smoker": "No", "day": "Sun", "time": "Dinner", "size": 3},
    {"total_bill": 21.01, "tip": 3.50, "sex": "Male", "smoker": "No", "day": "Sun", "time": "Dinner", "size": 3},
    {"total_bill": 23.68, "tip": 3.31, "sex": "Male", "smoker": "No", "day": "Sun", "time": "Dinner", "size": 2},
    {"total_bill": 24.59, "tip": 3.61, "sex": "Female", "smoker": "No", "day": "Sun", "time": "Dinner", "size": 4},
    {"total_bill": 25.29, "tip": 4.71, "sex": "Male", "smoker": "No", "day": "Sun", "time": "Dinner", "size": 4},
    {"total_bill": 8.77,  "tip": 2.00, "sex": "Male", "smoker": "No", "day": "Sun", "time": "Dinner", "size": 2},
    {"total_bill": 26.88, "tip": 3.12, "sex": "Male", "smoker": "No", "day": "Sun", "time": "Dinner", "size": 4},
    {"total_bill": 15.04, "tip": 1.96, "sex": "Male", "smoker": "No", "day": "Sun", "time": "Dinner", "size": 2},
    {"total_bill": 14.78, "tip": 3.23, "sex": "Female", "smoker": "No", "day": "Sun", "time": "Dinner", "size": 2},
])

# Create an Agent that uses the OpenAIAdapter as its LLM
agent = Agent(conn=None, llm=llm)

print("Running agent summarizer with OpenAI...")
result = agent.run("Summarize the dataframe contents and key statistics.", data=sample)

summ = result.get("execution", {}).get("results", {}).get("Summarizer")
print("\nSummarizer result:\n")
print(summ)
