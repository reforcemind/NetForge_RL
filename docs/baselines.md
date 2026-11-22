# Baselines

NetForge includes ready-to-run reference baselines for both defensive (Blue) and offensive (Red) roles:

- **`RandomPolicy`**: Samples uniformly from valid masked actions.
- **`HeuristicBluePolicy`**: Rule-based SOC defender prioritizing alerts by severity.
- **`HeuristicRedPolicy`**: Opportunistic uncoordinated attacker.
- **`KillChainRedPolicy`**: Multi-stage adversary executing reconnaissance, exploitation, and lateral movement.
- **`jax_ppo`**: Vectorized on-device IPPO policy trained on simulated telemetry.

For detailed benchmarks, training scripts, and evaluation tables, see [Benchmarks: Baselines](benchmarks/baselines.md).
