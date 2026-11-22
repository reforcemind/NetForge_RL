# Self-Play & Elo Tournament

In multi-agent security games, static baselines can lead to policy overfitting. NetForge provides an automated tournament system in `benchmarks/self_play.py` that computes Elo ratings across populations of Red and Blue agents.

## How Ratings Work

- **Shared Ladder**: Both Red and Blue agents share a unified rating scale initialized at 1000 Elo.
- **Match Scoring**:
  - Blue Match Score: Mean SLA uptime maintained across the episode ($\in [0, 1]$).
  - Red Match Score: $1 - \text{SLA}$ uptime ($\in [0, 1]$).
- **Tournament Scheduling**: Agents are paired in round-robin matchups across multiple random seeds and scenario topologies.

## Running a Tournament

Run a tournament from the command line:

```bash
python -m benchmarks.self_play --seeds 5 --max-ticks 150
```

Or execute via Python:

```python
from benchmarks.self_play import population_tournament

results = population_tournament(
    scenarios=["ransomware", "apt_espionage", "cloud_hybrid"],
    seeds=list(range(5)),
    max_ticks=150,
)
```

## Adding Custom Policies to the Pool

You can supply custom agent pools mapping agent names to factory functions:

```python
def make_my_agent(seed=0):
    return MyCustomPolicy(seed=seed)

custom_blue_pool = {
    "heuristic-blue": lambda seed: HeuristicBluePolicy(seed=seed),
    "my-blue-agent": make_my_agent,
}
```
