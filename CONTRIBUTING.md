# Contributing

NetForge is a cybersecurity RL gym. Prefer packs, metrics, wrappers, probes,
datasets, docs. MAPPO / QMIX stay in other repos. Blue trains on SIEM, not
ground-truth compromise.

[Code of Conduct](CODE_OF_CONDUCT.md).
[Discussions](https://github.com/reforcemind/NetForge_RL/discussions) if issues are locked.

- YAML pack
- Probe in `netforge_rl/arena/questions/`
- Reward variant
- HTML replay
- Capability probe

Issue → one PR from `main` → tests for behavior changes.

```bash
pip install -e ".[dev,docs]"
ruff check . && ruff format .
pytest tests/ -m fast
sphinx-build -b html -W --keep-going docs docs/_build/html
```

Gate: `scripts/verify_all.sh` or `scripts/verify_local.ps1`.
GitHub Actions (`ci.yml`) is ruff + pytest + Sphinx, and deploys GitHub Pages on `main`.
