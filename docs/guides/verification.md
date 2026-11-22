# Verification Guide

To maintain environment determinism and code quality, all contributions must pass the verification test suite before merging.

## Automated Verification Scripts

NetForge includes turnkey verification scripts for local development:

```bash
# Linux / macOS / Git Bash
./scripts/verify_all.sh

# Windows PowerShell
./scripts/verify_local.ps1
```

## Manual Verification Steps

You can run individual verification steps manually:

### 1. Code Formatting & Linting

```bash
ruff check .
ruff format --check .
```

### 2. Fast Test Suite

Run the fast unit and regression tests:

```bash
pytest tests/ -m fast
```

### 3. Parity & Replay Tests

Verify backend parity and deterministic observation replay:

```bash
pytest tests/parity/ -v
```

### 4. Documentation Build

Ensure Sphinx documentation builds with zero warnings:

```bash
sphinx-build -b html -W --keep-going docs docs/_build/html
```

## Continuous Integration

Every pull request triggers the automated GitHub Actions workflow (`.github/workflows/ci.yml`), which executes formatting checks, unit tests across supported Python versions, and builds documentation. On pushes to `main`, documentation is automatically deployed to GitHub Pages.
