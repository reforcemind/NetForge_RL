# Benchmarks Overview

The NetForge benchmark suite provides tools to measure simulation throughput, evaluate agent policies, run adversarial tournaments, and verify reproducibility.

## Key Components

- **[Baseline Policies](baselines.md)**: Standard reference agents including random sampling, rule-based heuristics, kill-chain adversaries, and vectorized JAX IPPO policies.
- **[Throughput Measurement](run.md)**: Execution speed benchmarks comparing standard Python execution with the vectorized JAX simulation kernel.
- **[Self-Play & Elo Ratings](self_play.md)**: Population tournaments that rank competing defensive and offensive policies on a unified Elo ladder.
- **[Research Probes](../questions.md)**: Pre-configured empirical experiments investigating belief-state observation, deception resistance, and out-of-distribution generalization.

## Quick CLI Execution

```bash
# Measure simulation throughput across batch sizes
python -m benchmarks.throughput --batches 1024 4096 --steps 50

# Run Elo self-play tournament
python -m benchmarks.self_play --seeds 5 --max-ticks 150

# Generate leaderboard rankings
python -m benchmarks.build_leaderboard --episodes 5 --max-steps 150
```
