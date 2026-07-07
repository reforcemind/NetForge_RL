<div align="center">
  <img src="https://img.shields.io/badge/version-3.0.0-blue?style=for-the-badge" alt="Version 3.0.0"/>
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/JAX-vmap%20%2B%20jit-orange?style=for-the-badge" alt="JAX"/>
  <img src="https://img.shields.io/badge/PettingZoo-MARL-purple?style=for-the-badge" alt="PettingZoo"/>
  <img src="https://img.shields.io/badge/MITRE_ATT%26CK-Mapped-red?style=for-the-badge" alt="MITRE ATT&CK"/>
</div>

# NetForge RL

**A fast, reproducible multi-agent reinforcement learning environment for autonomous
cyber-defense.** Red agents attack a procedurally generated enterprise/OT network
(exploit, pivot, escalate, ransomware, kinetic sabotage); Blue agents defend it (monitor,
analyze, isolate, deceive, restore). Built on the PettingZoo parallel API with a JAX
vectorized backend for high-throughput training.

---

## Why NetForge

- **Standardized API** — PettingZoo `ParallelEnv`, Gymnasium spaces, conformance-tested.
- **Partial observability that means something** — Blue sees only a filtered, optionally
  *delayed* SIEM feed; Red operates under fog of war and must recon before it can exploit.
- **Realistic telemetry** — actions emit real Windows/Sysmon event XML (4624, 4688, Sysmon
  1/3/10/22, …) which an NLP encoder turns into observations. Plug an LLM in as a SOC analyst.
- **MITRE ATT&CK aligned** — exploits map to real CVEs (MS17-010, CVE-2019-0708, Log4Shell).
- **Reproducible to the bit** — a fixed seed replays observations, SIEM embeddings, infos and
  rewards identically (test-guaranteed). See [Reproducibility](environment/reproducibility.md).
- **Tunable difficulty** — named `easy`/`medium`/`hard` presets and a frozen held-out
  evaluation split. See [Difficulty & Splits](environment/difficulty.md).
- **Diagnostic probes + capability cards** — a 6-capability suite (memory, attention, temporal,
  precision, safety, generalization) that isolates *what* a policy can and cannot do, summarised
  as a per-policy radar card. See [Diagnostics](diagnostics/overview.md).
- **Graph-native observations** — hosts as nodes, reachability as edges, fog-of-war aware,
  one line to PyTorch Geometric / jraph. See [Graph Observations](environment/graph_observations.md).
- **Self-play & Elo** — a population tournament that rates red and blue on one ladder. See
  [Self-Play](benchmarks/self_play.md).
- **Deception as a mechanic** — decoys and honeytokens with a `deception_efficacy` metric
  measuring how much of Red's effort was wasted on traps.
- **SOC export** — replay an episode as OCSF-style JSON for real SIEM tooling. See
  [SOC Export](interop/soc_export.md).
- **Trained baselines, not just scripted** — a JIT-fused JAX IPPO trainer whose whole rollout
  runs on-device; a committed `ransomware` run learns mean reward 0.06 → 0.71. Checkpoints
  reload into self-play and capability cards. See [Baselines](benchmarks/baselines.md).
- **Standard single-agent API** — `NetForgeSingleAgentEnv` is a `gymnasium.Env` (passes
  `check_env`) for Stable-Baselines3 / CleanRL, with the action mask in `info`.
- **MITRE ATT&CK coverage** — episodes report which ATT&CK techniques Red exercised and the
  fraction of the taxonomy covered (`attack_coverage`).
- **Fast** — JAX backend measured at **270,795 env-steps/s (1,083,181 agent-steps/s)** at
  batch 4096 on CPU, with an optional in-kernel numeric SIEM signal. See
  [Benchmarks](benchmarks/overview.md).

## Quick Start

```bash
pip install 'netforge_rl[jax] @ git+https://github.com/reforcemind/NetForge_RL'
```

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

See the [Quick Start Guide](quickstart.md) for the raw PettingZoo loop, the JAX vectorized
loop, difficulty presets, baselines, and diagnostics.

## What's inside

| Layer | Package | Purpose |
|---|---|---|
| Environment | `environment/` | PettingZoo `NetForgeRLEnv`, difficulty presets, curriculum |
| Scenarios | `scenarios/` | 5 reward/objective families (ransomware, APT, cloud, IoT, OT) |
| Actions | `actions/red`, `actions/blue` | ATT&CK-aligned red/blue capabilities |
| Telemetry | `siem/`, `nlp/` | Sysmon/Windows log synthesis + NLP encoding |
| JAX backend | `backends/jax/` | Vectorized `vmap`/`jit` kernels + NumPy reference |
| Baselines | `baselines/` | Random, heuristic, kill-chain red, JAX PPO |
| Diagnostics | `diagnostics/` | Capability probes + oracle information-asymmetry |
| Bridges | `bridges/` | RLlib, JaxMARL, CleanRL, DLPack adapters |
| Semantic | `semantic/` | LLM SOC agents, grammars, fine-tuning recipes |

## The five scenarios

| Scenario | Red objective | Blue objective | Terminal condition |
|---|---|---|---|
| `ransomware` | Encrypt/compromise Corporate + Secure | Contain, restore, avoid downtime | all Corporate/Secure compromised, or PLC kinetic |
| `apt_espionage` | Stealthy persistence + exfiltration | Detect and isolate every foothold | every infected host isolated |
| `cloud_hybrid` | Breach the Secure enclave | Protect the Secure subnet SLA | every Secure host compromised |
| `iot_grid` | Take the grid controllers | Keep controllers healthy | all controllers compromised |
| `ot_stuxnet` | Drive a PLC to kinetic destruction | Prevent physical damage | any PLC kinetic destruction |

## Citation

```bibtex
@misc{netforge_rl_2026,
  title  = {NetForge RL: A Fast Multi-Agent Cybersecurity Benchmark for Reinforcement Learning},
  author = {ReforceMind},
  year   = {2026},
  url    = {https://github.com/reforcemind/NetForge_RL}
}
```
