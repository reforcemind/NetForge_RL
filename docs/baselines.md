# Baselines

In-tree: `random`, `heuristic-blue`, `heuristic-red`, `killchain-red`,
JAX IPPO (`netforge_rl.baselines.jax_ppo`).

Your MAPPO / GNN / QMIX / LLM: train outside, evaluate here.

```bash
netforge benchmark --table --split dev --seeds 0 1 --max-ticks 40
```

Configs: `benchmarks/configs/`. Same split, seeds, and `max_ticks` when you compare runs.
