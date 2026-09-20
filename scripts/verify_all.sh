#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m ruff check .
python -m ruff format --check .
python -m pytest tests/ -m fast
python -m sphinx -b html -W --keep-going docs docs/_build/html
echo "ok"
