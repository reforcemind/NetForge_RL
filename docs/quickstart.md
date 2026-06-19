# Quick Start

Welcome to NetForge RL! This guide will take you from installation to running your first multi-agent cybersecurity simulation.

## 1. Installation

NetForge RL supports Python 3.14 and is distributed via pip. You can install the base package along with any optional backends you plan to use:

```bash
# Core installation with JAX, rendering, and LLM fine-tuning extras
pip install 'netforge_rl[jax,render,finetune] @ git+https://github.com/reforcemind/NetForge_RL'
```

## 2. Your First Simulation (PettingZoo API)

NetForge provides a multi-agent environment compatible with the PettingZoo API. Here is a basic 10-line loop using the legacy backend for the `ransomware` scenario:

```python
import numpy as np
from netforge_rl.environment.parallel_env import NetForgeRLEnv

# Initialize the environment
env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 50})
obs, _ = env.reset(seed=0)

while env.agents:
    # Random action sampling for each active agent
    actions = {a: np.array([0, 0], dtype=np.int64) for a in env.agents}
    
    # Step the environment
    obs, rewards, term, trunc, _ = env.step(actions)
    
    if all(term.values()) or all(trunc.values()):
        break

print("Simulation finished!")
```

## 3. High-Performance Vectorized Training (JAX)

For deep reinforcement learning pipelines, you need high throughput. NetForge's JAX backend can process millions of steps per second natively on CPU or GPU.

```python
import jax
from netforge_rl.backends.jax import VectorEnvSpec
from netforge_rl.bridges.jaxmarl import JaxMARLEnv, random_action_dict

# Initialize a hardware-accelerated environment with 4096 parallel universes
env = JaxMARLEnv(spec=VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3), batch_size=4096)
key = jax.random.PRNGKey(0)

# Reset and step fully on device
obs, state = env.reset(key)
obs, state, reward, done, info = env.step(key, state, random_action_dict(env, key))
```

## Next Steps

- **Red vs Blue Dynamics**: Read the [Cybersecurity Overview](cybersec/overview.md) to understand how the teams interact.
- **Ray RLlib**: Check out the [Training with RLlib](training/rllib.md) guide to hook NetForge into enterprise RL clusters.