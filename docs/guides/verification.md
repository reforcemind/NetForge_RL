# Verification

```bash
./scripts/verify_all.sh          # Linux / Git Bash
./scripts/verify_local.ps1       # Windows
```

```bash
pip install -e ".[dev,docs]"
ruff check . && ruff format --check .
pytest tests/ -m fast
sphinx-build -b html -W --keep-going docs docs/_build/html
```

CI: ruff + pytest + Sphinx HTML (`ci.yml`). Pages: `docs.yml` deploys to GitHub Pages on `main`.
Do not vendor MAPPO/QMIX/CT-GMARL.

{doc}`/getting-started` · {doc}`/python-quickstart` · {doc}`/contributing`
