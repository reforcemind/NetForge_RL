# NetForge Arena Evaluation

The NetForge Arena provides standardized evaluation splits, objective metrics, and adversarial populations to evaluate defensive reinforcement learning agents.

```{image} _static/figures/arena.svg
:alt: Arena evaluation splits
:class: fe-fig
```

## Evaluation Splits

Topologies, host vulnerability configurations, and seed spaces are partitioned into three evaluation tiers:

| Split | Visibility | Purpose | Details |
|---|---|---|---|
| `train` | Public | Policy Training | Standard scenario configurations with randomized initial seeds |
| `dev` | Public | Validation & Checkpointing | Frozen seed set (`EVAL_SEEDS`) with deterministic initial conditions |
| `hidden` | Held-out | Competition & Formal Benchmarks | Unseen network topologies and adversarial strategies |

## Multi-Metric Scoring

Rather than relying on scalar cumulative rewards (which can mask reckless strategies like isolating the entire network), the Arena tracks composite operational metrics:

- **SLA Uptime Percentage**: The percentage of healthy, operational hosts kept online throughout the episode.
- **Mean Time to Containment (MTTC)**: Average ticks required to isolate or remediate a compromised host.
- **False Positive Rate (FPR)**: Rate of defensive isolation actions applied to uncompromised machines.
- **Exfiltrated Data**: Total megabytes of confidential data successfully exfiltrated by Red.
- **Kinetic Destruction**: Any catastrophic physical failure in OT/SCADA scenarios (hard failure condition).
- **Conditional Value at Risk (CVaR)**: Tail-risk metric quantifying policy performance in the worst 5% of evaluation episodes.

## Running Evaluations

Evaluate an agent policy from the command line:

```bash
# Evaluate a submission script on dev split
netforge evaluate examples/submissions/heuristic_blue.py --seeds 0 1 --max-ticks 40

# Run full Arena benchmark table
netforge benchmark --split dev --table --seeds 0 1 --max-ticks 40
```
