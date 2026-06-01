# NetForge Baselines Leaderboard

Reference numbers for the three shipped baselines (`random`,
`heuristic-blue`, `heuristic-red`) against every scenario. Reproduce with:

```bash
python -m benchmarks.build_leaderboard --episodes 5 --max-steps 80
```

Files:
- [`baselines.json`](baselines.json) — raw per-episode results in the same
  schema as the Phase 8 zero-shot LLM leaderboard.
- [`baselines_summary.json`](baselines_summary.json) — per-(model, scenario)
  aggregates, sorted by `mean_total_reward` descending.

Schema (`baselines.json` entries):

| Field | Meaning |
|---|---|
| `model_id` | policy name |
| `scenario` | env scenario_type |
| `steps` | ticks elapsed before termination/truncation |
| `rewards` | per-agent cumulative reward dict |
| `final_compromised` | host count owned by Red at episode end |
| `final_isolated` | host count in `isolated` state at end (includes padding) |
| `invalid_replies` | parser failures (zero for non-LLM baselines) |

The harness produces leaderboard JSON identical to
`netforge_rl.semantic.run_episode` output so LLM and classical baselines
share one report format. Mix them with `summarize(path)`.
