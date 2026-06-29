# Technical Datasheet

## Environment Composition
- **Topology Size**: 100 host array slots. Active topologies range from 15-30 nodes. Unused nodes use `169.254.0.0/16` addresses, serving as dynamic decoys.
- **State Representation**: `GlobalNetworkState` contains host attributes, firewall ACLs, topologies, and visibility masks. Categorical properties are encoded against static codebooks (`netforge_rl/core/functional.py`).
- **Scenarios**: `ransomware`, `apt_espionage` (v2.2.0).

## Metrics and Reward Signals
- Reward signals are defined per-scenario via `BaseScenario`. Rewards are `tanh`-normalized.
- Metrics emitted per episode:
  - Mean Time To Containment (MTTC)
  - SLA Uptime
  - Total Exfiltrated Data Volume
  - Host Compromise/Isolation Counts

## Data Generation
- **Source**: `netforge_rl/topologies/network_generator.py`.
- **Determinism**: Deterministic based on seed.
- **Splits**: Leaderboard evaluation uses seeds 0-49.
- **Limitations**: Synthetic generation.

## Supported Use Cases
- RL policy training (PPO, MAPPO).
- LLM zero-shot evaluation and PEFT fine-tuning.
