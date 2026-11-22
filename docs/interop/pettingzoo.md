# PettingZoo & Gymnasium Interfaces

NetForge adheres strictly to Farama Foundation standards, implementing both [PettingZoo](https://pettingzoo.farama.org/) (for multi-agent training) and [Gymnasium](https://gymnasium.farama.org/) (for single-agent training).

## Multi-Agent PettingZoo

NetForge provides both parallel and turn-based (AEC) PettingZoo environments.

### Parallel API

In the Parallel API, all agents choose and submit actions simultaneously at each step:

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

### Agent Roster

Multi-agent scenarios feature up to four active roles:

- `red_operator`: Adversary conducting scanning, exploitation, and lateral movement.
- `blue_dmz`: Perimeter defender responsible for web and proxy tier hosts.
- `blue_internal`: Corporate defender managing workstation and file-share tiers.
- `blue_restricted`: High-security defender guarding databases, domain controllers, and OT controllers.

## Single-Agent Gymnasium

To benchmark single-agent defense algorithms (e.g., CleanRL, Stable-Baselines3) against automated adversaries, NetForge provides a Gymnasium wrapper:

```python
import gymnasium as gym
import netforge_rl

env = gym.make("NetForge/Blue-v4", max_ticks=100)
obs, info = env.reset(seed=42)

for _ in range(100):
    mask = info.get("action_mask")
    action = env.action_space.sample(mask=mask)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```
