<div align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/JAX-vmap%20%2B%20jit-orange?style=for-the-badge" alt="JAX"/>
  <img src="https://img.shields.io/badge/PettingZoo-MARL-purple?style=for-the-badge" alt="PettingZoo"/>
  <img src="https://img.shields.io/badge/MITRE_ATT%26CK-Mapped-red?style=for-the-badge" alt="MITRE ATT&CK"/>
  <img src="https://img.shields.io/badge/LLM%20policies-trl%20%2B%20LoRA-green?style=for-the-badge" alt="LLM policies"/>
</div>

<h1 align="center">NetForge RL</h1>

<p align="center">
  <a href="ROADMAP.md">Roadmap</a> · <a href="AUDIT.md">Audit</a> · <a href="changelog.md">Changelog</a> · <a href="notebooks/">Notebooks</a>
</p>

---

## What's new in v4.0

- **JAX backend**: `jax.vmap` + `jax.jit` vectorized step delivering **1,082,255 aggregate SPS at 4096 parallel envs on CPU** — 103,500× the single-env legacy baseline. GPU/TPU pushes this further.
- **JaxMARL-shape API + DLPack zero-copy**: ride the JAX rollouts from CleanRL / Stable-Baselines3 / RLlib without leaving the device.
- **Decoupled renderer**: `env.render('rgb_array')` produces matplotlib + NetworkX frames off the hot path; `FrameRecorder` writes mp4 / gif via moviepy.
- **Semantic Bridge** (Phase 8): the first MARL benchmark in which language and vision models are first-class agents — `state_to_text` SIEM reports, `build_vla_prompt` multimodal payloads, a regex action parser with grammar-constrained-decoding hooks, a zero-shot leaderboard harness, and a `trl`-based LoRA + PPO recipe that fine-tunes Llama-3-8B on a single 24 GB GPU.
- **Functional core**: frozen `EnvState` PyTree with parity-tested pure `apply_state_delta` and `resolve_conflicts`. Legacy backend is byte-identical to the Phase 0 baseline (golden trajectory hash `abd164a5…` preserved across every refactor).

## Why NetForge

High-fidelity MARL cybersecurity simulators (CybORG / CAGE) run at hundreds of steps per second and resist large-scale experimentation. Throughput-optimized envs (JaxMARL, SMAX) lack the partial observability, SIEM telemetry, MITRE ATT&CK alignment, zero-trust enforcement, and OT/ICS kinetic impact the security research community actually needs.

NetForge-MARL resolves the trade-off through a **dual-backend** architecture sharing one functional core, and is further the first cybersecurity MARL benchmark in which **foundation models are first-class agents** alongside PPO/MAPPO/QMIX.

## Quick start

```bash
pip install 'netforge_rl[jax,render,finetune] @ git+https://github.com/reforcemind/NetForge_RL'
```

### Legacy PyTorch backend (10 lines)

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

### JAX vectorized backend (4096 envs)

```python
import jax
from netforge_rl.backends.jax import VectorEnvSpec
from netforge_rl.bridges.jaxmarl import JaxMARLEnv, random_action_dict

env = JaxMARLEnv(spec=VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3), batch_size=4096)
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)
obs, state, reward, done, info = env.step(key, state, random_action_dict(env, key))
```

### Run an LLM as a SOC operator

```python
from netforge_rl.semantic import state_to_text, parse_action
prompt = state_to_text(env.to_envstate(), agent_id='blue_dmz')
# ...send prompt to your favorite API, then:
action_idx = parse_action(model_reply, 'blue_dmz', sorted(env.global_state.all_hosts))
```

### Fine-tune Llama-3-8B with LoRA + PPO

See [`notebooks/07_finetune_llama3_lora.ipynb`](notebooks/07_finetune_llama3_lora.ipynb). Targets a 4-hour wall-clock on a single 24 GB GPU.

## Architecture

```
netforge_rl/
├── core/functional.py         frozen EnvState PyTree + pure interpreters
├── backends/jax/              vmap'd kernels + vector_env (≥10⁶ SPS @ 4096)
├── environment/parallel_env   legacy PettingZoo backend (golden hash locked)
├── bridges/{jaxmarl,dlpack}   JaxMARL adapter + zero-copy torch↔jax
├── render/                    decoupled matplotlib + recorder (off hot path)
├── semantic/                  LA + VLA wrappers, zero-shot runner, leaderboard,
│   └── finetune/                trl LoRA+PPO adapter
├── baselines/                 RandomPolicy, HeuristicBlue/Red, JAX PPO
└── scenarios/                 ransomware, APT espionage (+ Phase 5 suite expanding)
```

## Citation

If you use NetForge RL, please cite:

```bibtex
@misc{netforge_rl_2026,
  title  = {NetForge RL},
  author = {ReforceMind},
  year   = {2026},
  url    = {https://github.com/reforcemind/NetForge_RL}
}
```
