# Quick Start Guide

## Installation

Requires Python 3.14.

```bash
pip install 'netforge_rl[jax,render,finetune] @ git+https://github.com/reforcemind/NetForge_RL'
```

## Legacy PettingZoo Execution

Execution loop using the legacy unvectorized Python backend.

```python
import numpy as np
from netforge_rl.environment.parallel_env import NetForgeRLEnv

env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 50})
obs, _ = env.reset(seed=0)

while env.agents:
    actions = {a: np.array([0, 0], dtype=np.int64) for a in env.agents}
    obs, rewards, term, trunc, _ = env.step(actions)
    
    if all(term.values()) or all(trunc.values()):
        break
```

## JAX Vectorized Execution

Hardware-accelerated batched execution loop.

```python
import jax
from netforge_rl.backends.jax import VectorEnvSpec
from netforge_rl.bridges.jaxmarl import JaxMARLEnv, random_action_dict

env = JaxMARLEnv(spec=VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3), batch_size=4096)
key = jax.random.PRNGKey(0)

obs, state = env.reset(key)
obs, state, reward, done, info = env.step(key, state, random_action_dict(env, key))
```