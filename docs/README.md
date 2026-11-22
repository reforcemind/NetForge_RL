# Documentation Guide

This directory contains the Sphinx documentation source files for NetForge.

## Building the Documentation

1. Install documentation dependencies:
   ```bash
   pip install -e ".[docs]"
   ```

2. Build HTML documentation:
   ```bash
   sphinx-build -b html -W --keep-going docs docs/_build/html
   ```

3. Open `docs/_build/html/index.html` in your browser.

## Documentation Structure

- `docs/getting-started.md`: Installation and introductory walkthrough.
- `docs/python-quickstart.md`: Python API usage (Gymnasium & PettingZoo).
- `docs/scenarios.md`: Catalog of attack scenarios and reward modes.
- `docs/cybersec/`: Threat model, Red actions, Blue defenses.
- `docs/architecture/`: Tick loop, action queues, OT physics, SIEM logging.
- `docs/benchmarks/`: Benchmark suites, throughput tests, baseline policies, Elo ratings.
- `docs/environment/`: Graph observations, difficulty presets, determinism.
- `docs/api/`: Autodoc API references.
