# Baselines

`netforge_rl.baselines`.

| Policy | |
|---|---|
| `RandomPolicy` | uniform `MultiDiscrete([32, 100])` |
| `HeuristicBluePolicy` | isolate first compromised host, else analyse |
| `HeuristicRedPolicy` | exploit without recon — usually 0 compromises |
| `KillChainRedPolicy` | recon → exploit → pivot; ~2–3 hosts/ep, SLA drops |
| `jax_ppo` | on-device IPPO (`baselines/jax_ppo.py`) |

`ExploitRemoteService` needs prior `DiscoverNetworkServices`. Use kill-chain as
the Red reference, not heuristic-red.

```bash
python -m benchmarks.build_leaderboard --episodes 5 --max-steps 150
python -m benchmarks.train_curve --name blue_ransomware --iters 40
```

Committed IPPO on ransomware: mean reward **0.06 → 0.71** (40 iters, CPU).
Load with `jax_ppo.load_params`. RLlib: `benchmarks/rllib_rmappo.py`.
{doc}`run` · {doc}`self_play`
