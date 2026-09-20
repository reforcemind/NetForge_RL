# Docs map

| | |
|---|---|
| First episode | `README.md`, {doc}`getting-started` |
| Env ids | {doc}`python-quickstart` |
| Red / Blue | {doc}`cybersec/overview` |
| Package | `netforge_rl/` |
| Figures | `docs/_static/figures/*.svg` + `*.excalidraw` |

```bash
python docs/_static/figures/build.py
pip install -e ".[docs]"
sphinx-build -b html -W --keep-going docs docs/_build/html
```
