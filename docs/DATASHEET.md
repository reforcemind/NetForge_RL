# Datasheet

| | |
|---|---|
| Slots | 100 hosts; 15–30 live; padding `169.254.0.0/16` (decoys) |
| State | `GlobalNetworkState` + codebooks in `core/functional.py` |
| Families | `ransomware`, `apt_espionage`, `cloud_hybrid`, `iot_grid`, `ot_stuxnet` |
| Agents | `red_operator`, `blue_dmz`, `blue_internal`, `blue_restricted` |
| Action | `MultiDiscrete([32, 100])`, mask `int8[132]` |
| Obs | `obs` 256, `siem_embedding` 128, `action_mask` 132, `adj_matrix`, `delta_t`; Blue `blue_comm` 100 |
| Spec | `python -m benchmarks.env_spec --json` |
| Difficulty | `easy` / `medium` / `hard`; `evaluation_mode` seed offset 1000; `EVAL_SEEDS` |
| Reward | per-scenario `REWARD_WEIGHTS`, `tanh`-normalized |
| Metrics | MTTC, SLA, exfil, detection, compromise/isolation (padding excluded) |
| Replay | seeded obs / SIEM / infos / rewards; fixed epoch, not wall-clock |
| Limits | synthetic; no real captures |

Also: belief graphs, decoys, OCSF export, Elo self-play, JAX telemetry, IPPO
checkpoints, `NetForge/Blue-v4`, ATT&CK `attack_coverage`.
