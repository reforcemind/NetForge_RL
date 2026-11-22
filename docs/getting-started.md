# Getting Started

NetForge requires Python 3.12 or newer.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/reforcemind/NetForge_RL.git
cd NetForge_RL
pip install -e .
```

To install development dependencies (testing and documentation tools):

```bash
pip install -e ".[dev,docs]"
```

## Running Your First Simulation

NetForge includes a command-line interface for running scenarios and inspecting results:

```bash
netforge run hospital_ransomware --replay replay.html
```

This runs a simulation of a ransomware outbreak targeting a clinical network, pitting a Red adversary against Blue defensive policies. When finished, it outputs `replay.html`, an interactive visualizer that displays network topology, host compromise status, SIEM alerts, and reward timelines across every tick.

## Python Quickstart

### Gymnasium Interface

For standard single-agent reinforcement learning (controlling Blue against a scripted adversary):

```python
import gymnasium as gym
import netforge_rl

env = gym.make("NetForge/Blue-v4", max_ticks=100)
obs, info = env.reset(seed=42)

for tick in range(100):
    action = env.action_space.sample(mask=info.get("action_mask"))
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### PettingZoo Interface

For multi-agent reinforcement learning with discrete defensive tiers (`blue_dmz`, `blue_internal`, `blue_restricted`) and an active adversary (`red_operator`):

```python
import netforge_rl
from pettingzoo import make

env = make("parallel", "netforge/ransomware-v4", max_ticks=100)
obs, infos = env.reset(seed=42)

while env.agents:
    actions = {
        agent: env.action_space(agent).sample(mask=infos[agent].get("action_mask"))
        for agent in env.agents
    }
    obs, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

## Next Steps

- Explore the [Python API Guide](python-quickstart.md) for action masks and observation shapes.
- Learn about [Cyber Mechanics](cybersec/overview.md) and the threat model.
- Browse available [Scenario Packs](scenarios.md).
- Evaluate custom policies with the [Evaluation Suite](arena.md).
