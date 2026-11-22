# Baseline Policies

NetForge includes reference baseline agents in `netforge_rl.baselines`:

| Policy Name | Team | Description |
|---|---|---|
| `RandomPolicy` | Red / Blue | Uniformly samples legal actions from `MultiDiscrete([32, 100])` using the action mask |
| `HeuristicBluePolicy` | Blue | Prioritizes isolating hosts with detected high-severity SIEM alerts; performs triage analysis otherwise |
| `HeuristicRedPolicy` | Red | Naive opportunistic attacker attempting exploits without service reconnaissance |
| `KillChainRedPolicy` | Red | Structured adversary executing disciplined reconnaissance, exploit chaining, privilege escalation, and lateral movement |
| `jax_ppo` | Blue | Independent PPO policy trained on the vectorized JAX environment (`baselines/jax_ppo.py`) |

## Evaluation Recommendations

When evaluating defensive policies:

- **Reference Adversary**: Always benchmark against `KillChainRedPolicy` rather than naive heuristics. The kill-chain adversary systematically discovers network services before attempting exploits, generating realistic multi-stage intrusion traffic.
- **Trained IPPO Checkpoint**: NetForge ships with a pre-trained JAX IPPO checkpoint on `ransomware-v4` achieving strong mean rewards over baseline runs. Checkpoints can be loaded via `jax_ppo.load_params`.

## Benchmarking Baselines

```bash
# Benchmark heuristic baselines across dev split
netforge benchmark --split dev --seeds 0 1 --max-ticks 40

# Train JAX IPPO baseline curve
python -m benchmarks.train_curve --name blue_ransomware --iters 40
```
