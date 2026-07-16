<div align="center">
  <img src="https://img.shields.io/badge/version-3.1.0-blue?style=for-the-badge" alt="Version 3.1.0"/>
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/JAX-vmap%20%2B%20jit-orange?style=for-the-badge" alt="JAX"/>
  <img src="https://img.shields.io/badge/PettingZoo-MARL-purple?style=for-the-badge" alt="PettingZoo"/>
  <img src="https://img.shields.io/badge/MITRE_ATT%26CK-Mapped-red?style=for-the-badge" alt="MITRE ATT&CK"/>
</div>

<h1 align="center">NetForge RL</h1>

<p align="center">Multi-agent cybersecurity environment for RL research. Red team vs Blue team on generated networks.</p>

---

## Why NetForge

- **Standardized API** — PettingZoo `ParallelEnv`, Gymnasium spaces, conformance-tested.
  A `gymnasium.Env` single-agent facade (`NetForgeSingleAgentEnv`) drops straight into
  Stable-Baselines3 or CleanRL.
- **Partial observability that means something** — Blue reads a filtered, optionally
  *delayed* SIEM feed; Red must recon a host before it can exploit it.
- **Real telemetry** — actions emit real Windows/Sysmon event XML, encoded into
  observations by an NLP pipeline. Point an OpenAI, Anthropic, Google, or vLLM client at
  the raw logs and let it play SOC analyst.
- **MITRE ATT&CK aligned** — actions carry ATT&CK technique IDs, and CVE identifiers are
  used as abstract vulnerability labels (no real exploit code); every episode reports which
  ATT&CK techniques Red actually exercised.
- **Reproducible to the bit** — a fixed seed replays observations, SIEM embeddings, infos,
  and rewards identically. Test-guaranteed, not just claimed.
- **Tunable difficulty** — named `easy` / `medium` / `hard` presets plus a frozen held-out
  evaluation split, so results are comparable across runs.
- **Diagnostic probes + capability cards** — 6 targeted probes (memory, attention,
  temporal, precision, safety, generalization) score *what* a policy can and can't do,
  summarized as a per-policy radar chart.
- **Deception as a mechanic** — decoys and honeytokens with a `deception_efficacy` metric
  quantifying how much of Red's effort was wasted on traps.
- **Optional graph observations** — an experimental wrapper exposes hosts as nodes and
  reachability as edges (fog-of-war aware); the benchmarked default is the fixed-shape array.
- **Self-play & Elo** — a population tournament rates every red and blue policy on one
  shared Elo ladder.
- **Fast** — the JAX backend reaches **~2.5×10⁵ env-steps/s (~1.0M agent-steps/s)** at batch
  4096 on CPU. It runs a reduced transition core with an in-kernel scalar alert (not the full
  SIEM text pipeline), so it is a throughput surrogate, not a replica of the Python engine.
- **Trained, not just scripted, baselines** — a JIT-fused IPPO trainer whose entire
  rollout runs on-device; a committed run learns mean reward 0.06 → 0.71 on `ransomware`.

## Install

```bash
pip install 'netforge_rl[jax] @ git+https://github.com/reforcemind/NetForge_RL'
```

Extras: `jax` (vectorized backend), `render` (visualization), `finetune` (LLM PEFT),
`rllib` (Ray multi-agent training). See the
[install guide](https://reforcemind.github.io/NetForge_RL/quickstart/) for the full list.

## Quick start

```python
from netforge_rl.environment import make_env

env = make_env('medium', scenario_type='ransomware', seed=0)
obs, infos = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, term, trunc, infos = env.step(actions)
    if all(term.values()) or all(trunc.values()):
        break
```

### JAX vectorized (4096 environments)

```python
import jax
from netforge_rl.backends.jax import VectorEnvSpec
from netforge_rl.bridges.jaxmarl import JaxMARLEnv, random_action_dict

env = JaxMARLEnv(spec=VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3), batch_size=4096)
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)
obs, state, reward, done, info = env.step(key, state, random_action_dict(env, key))
```

### LLM as a Blue agent

```python
from netforge_rl.semantic import state_to_text, parse_action

prompt = state_to_text(env.to_envstate(), agent_id='blue_dmz')
# send prompt to your model, then:
action_idx = parse_action(model_reply, 'blue_dmz', sorted(env.global_state.all_hosts))
```

See the [Quick Start Guide](https://reforcemind.github.io/NetForge_RL/quickstart/) for
difficulty presets, baselines, diagnostics, and the Gymnasium single-agent wrapper.

## The five scenarios

| Scenario | Red objective | Blue objective | Terminal condition |
|---|---|---|---|
| `ransomware` | Compromise Corporate + Secure | Contain, restore, avoid downtime | all Corporate/Secure compromised, or PLC kinetic |
| `apt_espionage` | Stealthy persistence + exfiltration | Detect and isolate every foothold | every infected host isolated |
| `cloud_hybrid` | Breach the Secure enclave | Protect the Secure subnet SLA | every Secure host compromised |
| `iot_grid` | Take the grid controllers | Keep controllers healthy | all controllers compromised |
| `ot_stuxnet` | Drive a PLC to kinetic destruction | Prevent physical damage | any PLC kinetic destruction |

## Package layout

| Package | Purpose |
|---|---|
| `environment/` | PettingZoo `NetForgeRLEnv`, difficulty presets, curriculum, Gymnasium facade |
| `scenarios/` | 5 reward/objective families sharing one `BaseScenario` |
| `actions/red`, `actions/blue` | ATT&CK-aligned capabilities + technique mapping |
| `siem/`, `nlp/` | Sysmon/Windows log synthesis, NLP encoding, OCSF-style export |
| `backends/jax/` | Vectorized `vmap`/`jit` transition kernel + NumPy reference |
| `baselines/` | Random, heuristic, kill-chain red, JAX IPPO |
| `diagnostics/` | Capability probes, oracle information-asymmetry, capability cards |
| `bridges/` | RLlib, JaxMARL, CleanRL, DLPack adapters |
| `semantic/` | LLM SOC agents, prompt grammars, fine-tuning recipes |

## Citation

```bibtex
@misc{jankowski2026netforgerlmultiagentsimulation,
      title={NetForge RL: A Multi-Agent Simulation Environment for Cyber Defense with Durative Actions}, 
      author={Igor Jankowski},
      year={2026},
      eprint={2604.09523},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.09523}, 
}
```

## License

MIT — see [LICENSE](LICENSE).
