$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m ruff check .
python -m ruff format --check .
python -m pytest tests/ -m fast
python -m sphinx -b html -W --keep-going docs docs/_build/html
Write-Host "ok"
