<div align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&cachebust=1" alt="Python 3.14"/>
  <img src="https://img.shields.io/badge/JAX-vmap%20%2B%20jit-orange?style=for-the-badge" alt="JAX"/>
  <img src="https://img.shields.io/badge/PettingZoo-MARL-purple?style=for-the-badge" alt="PettingZoo"/>
  <img src="https://img.shields.io/badge/MITRE_ATT%26CK-Mapped-red?style=for-the-badge" alt="MITRE ATT&CK"/>
  <img src="https://img.shields.io/badge/LLM%20agents-trl%20%2B%20LoRA-green?style=for-the-badge" alt="LLM agents"/>
</div>

<h1 align="center">NetForge RL</h1>

<p align="center">Multi-agent cybersecurity environment for RL research. Red team vs Blue team on generated networks.</p>

---

## What is it

A MARL environment where Red agents attack a network (exploit, pivot, ransomware) and Blue agents defend it (isolate, patch, monitor). Built on PettingZoo with a JAX backend for fast vectorized training.

- **Partial observability**: Blue sees SIEM logs. Red has fog of war.
- **Realistic telemetry**: Actions generate real Windows/Sysmon event strings.
- **MITRE ATT&CK aligned**: Exploits map to real CVEs (MS17-010, CVE-2019-0708).
- **LLM agents**: Plug in GPT-4 or Llama3 as a SOC operator reading raw logs.
- **JAX backend**: 1M+ steps/sec at 4096 parallel envs on CPU.

## Install

```bash
pip install 'netforge_rl[jax] @ git+https://github.com/reforcemind/NetForge_RL'
```

## Documentation

is available at:
**[https://reforcemind.github.io/NetForge_RL/](https://reforcemind.github.io/NetForge_RL/)**

## Usage

```python
from netforge_rl.environment.parallel_env import NetForgeRLEnv
import numpy as np

env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 50})
obs, _ = env.reset(seed=0)
while env.agents:
    actions = {a: np.array([0, 0], dtype=np.int64) for a in env.agents}
    obs, rewards, term, trunc, _ = env.step(actions)
```

### JAX vectorized (4096 envs)

```python
import jax
from netforge_rl.backends.jax import VectorEnvSpec
from netforge_rl.bridges.jaxmarl import JaxMARLEnv, random_action_dict

env = JaxMARLEnv(spec=VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3), batch_size=4096)
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)
obs, state, reward, done, info = env.step(key, state, random_action_dict(env, key))
```

### LLM as Blue agent

```python
from netforge_rl.semantic import state_to_text, parse_action

prompt = state_to_text(env.to_envstate(), agent_id='blue_dmz')
# send prompt to your API, then:
action_idx = parse_action(model_reply, 'blue_dmz', sorted(env.global_state.all_hosts))
```

## Citation

```bibtex
@misc{netforge_rl_2026,
  title  = {NetForge RL},
  author = {ReforceMind},
  year   = {2026},
  url    = {https://github.com/reforcemind/NetForge_RL}
}
```
