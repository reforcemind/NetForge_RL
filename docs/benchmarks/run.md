# Benchmark Execution & Metrics

This guide details how to benchmark simulation speed, run competition evaluations, and interpret diagnostic metrics.

## Simulation Throughput

NetForge features both a standard Python environment (optimizing for rich SIEM telemetry and graph observations) and a vectorized JAX kernel (optimizing for high-throughput rollout generation).

Measure throughput on your local machine:

```bash
python -m benchmarks.throughput --batches 1024 4096 --steps 50 --warmup 2
```

Representative throughput performance on modern CPU hardware:

| Backend | Batch Size | Env-Steps / Second | Agent-Steps / Second |
|---|---|---|---|
| **Python** | 1 | ~300 | ~1,200 |
| **JAX** | 1,024 | ~158,000 | ~634,000 |
| **JAX** | 4,096 | ~270,000 | ~1,080,000 |

## Competition Evaluation

To generate competitive leaderboard metrics across agents:

```bash
# Run competition evaluation for a specific agent
python -m benchmarks.competition_eval --name my_blue_agent --team blue --episodes 10

# Generate leaderboard summary
python -m benchmarks.build_leaderboard --episodes 5 --max-steps 150
```

Composite scores balance security efficacy and network availability:

- **Blue Score**: $\text{SLA} \times 50 - \text{Compromised} \times 2 - \text{MTTC} \times 0.1 + \text{Reward} \times 0.1$
- **Red Score**: $\text{Compromised} \times 2 + \text{Exfiltrated} \times 0.01 - \text{SLA} \times 10 + \text{Reward} \times 0.1$

## Step & Episode Metrics Dictionary

The `info` dictionary provides granular tracking at each step:

| Metric Key | Description |
|---|---|
| `compromised_hosts` | Count of currently compromised active hosts (padding slots excluded) |
| `isolated_hosts` | Count of currently isolated active hosts |
| `active_hosts` | Total active hosts in the scenario topology |
| `SLA_Uptime_Percentage` | Fraction of healthy online hosts over time |
| `MTTC` / `containment_time` | Average ticks elapsed between compromise and defensive containment |
| `detection_rate` | Ratio of compromised hosts successfully identified and isolated |
| `Total_Exfiltrated_Data` | Cumulative data exfiltrated by Red (MB) |
| `false_positives` | Count of uncompromised hosts isolated by Blue |
| `deception_hits` | Adversary actions that landed on decoys or honeytokens |
| `attack_coverage` | Percentage of MITRE ATT&CK techniques observed in the run |
