# Agent notes

Verification is local. GitHub Actions (`ci.yml`) runs ruff + pytest + Sphinx
and deploys GitHub Pages on push to `main`.

- Full gate: `scripts/verify_all.sh` (Linux/Git Bash) or `scripts/verify_local.ps1` (Windows)
- User CLI: `python -m netforge` / `netforge`
- Docs: `sphinx-build -b html -W --keep-going docs docs/_build/html`
- Contract: `docs/guides/verification.md`

Sister project docs kit: [FlowEdge](https://github.com/reforcemind/FlowEdge).
Do not vendor MAPPO/QMIX/CT-GMARL into this tree.
