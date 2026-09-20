# Run

```bash
python -m benchmarks.throughput --batches 1024 4096 --steps 50 --warmup 2
```

CPU. `agent-steps/s = env-steps/s × agents`.

| Backend | Batch | env-steps/s | agent-steps/s |
|---|---|---|---|
| Python | 1 | ~300 | ~1,200 |
| JAX | 1024 | 158,540 | 634,160 |
| JAX | 4096 | 270,795 | 1,083,181 |

## Competition

`benchmarks/competition_eval.py` → `results/leaderboard.json`.

Blue: `SLA×50 − compromised×2 − MTTC×0.1 + blue_reward×0.1`  
Red: `compromised×2 + exfil×0.01 − SLA×10 + red_reward×0.1`

```bash
python -m benchmarks.competition_eval --name my_agent --team blue --episodes 10
python -m benchmarks.run_benchmark --name heuristic --team blue --seeds 20
python -m benchmarks.run_benchmark --name killchain --team red --red killchain --blue heuristic --seeds 20
python -m benchmarks.run_benchmark --name killchain --team red --red killchain --gap --seeds 20
python -m benchmarks.build_leaderboard --episodes 5 --max-steps 150
python -m benchmarks.env_spec --json
```

| `info` key | |
|---|---|
| `compromised_hosts` / `isolated_hosts` / `active_hosts` | padding excluded |
| `SLA_Uptime_Percentage` | healthy online fraction |
| `MTTC` / `containment_time` | time to contain |
| `detection_rate` | compromises that were isolated |
| `Total_Exfiltrated_Data` | cumulative |
| `false_positives` / `successful_exploits` / `services_restored` | step outcomes |
| `deception_hits` / `deception_efficacy` | decoy / honeytoken |
| `attack_techniques` / `attack_coverage` | ATT&CK ids |
| `blue_score` / `red_score` / `normalized_reward` | composites |
| `agent_energy` | remaining budget |
| `information_asymmetry` | under `DiagnosticsWrapper` |
| `__curriculum__` | phase / mean_reward / … |

{doc}`baselines`
