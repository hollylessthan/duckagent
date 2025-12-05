#!/usr/bin/env bash
set -euo pipefail

# Run tests for duckagent project in a reproducible venv
# Usage: ./scripts/run_tests.sh [pytest-args]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

# Install package editable so tests can import the src package
pip install -e .

# Run pytest with any forwarded args
pytest -q "$@"
