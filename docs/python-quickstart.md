# Python

`import netforge_rl` registers the ids. CPython 3.12+. Not on PyPI.

```bash
python -m pip install 'netforge-rl @ git+https://github.com/reforcemind/NetForge_RL'
# from a clone:
python -m pip install -e ".[dev]"
```

```python
import netforge_rl
import gymnasium as gym
from pettingzoo import make

env = make("parallel", "netforge/ransomware-v4", max_ticks=200)
obs, infos = env.reset(seed=0)
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, term, trunc, infos = env.step(actions)

blue = gym.make("NetForge/Blue-v4", max_ticks=200)
```

| API | Id | |
|---|---|---|
| PettingZoo parallel | `netforge/ransomware-v4` | 1 Red + 3 Blue |
| PettingZoo parallel | `netforge/arena-v4` | same factory |
| PettingZoo AEC | same ids | turn-based |
| Gymnasium | `NetForge/Blue-v4` | one Blue vs scripted Red |
| Gymnasium | `NetForge-v4` | alias |

Action: `MultiDiscrete([32, 100])`, mask `int8[132]`.
Blue graph: `GraphObservationWrapper`.

| Extra | |
|---|---|
| *(none)* | env, CLI, heuristics |
| `dev` | ruff, pytest |
| `docs` | Sphinx |
| `jax` | vectorized surrogate |
| `rllib` | RLlib env wrapper |
| `offline` | HDF5 export |
| `nlp` / `llm_clients` | SIEM encoder / LLM agents |

{doc}`interop/pettingzoo`
