<div align="center">
<img src="docs/_static/figures/mark.svg" width="96" alt="NetForge">

# NetForge

**Cybersecurity gym for reinforcement learning**

Red attacks a network. Blue contains it from SIEM, not from an oracle.
PettingZoo (1 Red + 3 Blue) or Gymnasium (one Blue vs scripted Red).

**Python 3.12+** · **PettingZoo** · **Gymnasium** · **Not on PyPI**

[Docs](https://reforcemind.github.io/NetForge_RL/) ·
[Start](https://reforcemind.github.io/NetForge_RL/getting-started.html) ·
[Python](https://reforcemind.github.io/NetForge_RL/python-quickstart.html) ·
[Cyber](https://reforcemind.github.io/NetForge_RL/cybersec/overview.html) ·
[Contributing](CONTRIBUTING.md)

<p>
  <img src="docs/_static/figures/purpose.svg" width="920" alt="Red, Blue, and the network">
</p>
</div>

## 📰 News

**🗓️ September 2026**

- 🛡️ **20** — Gym ids: PettingZoo `netforge/ransomware-v4` (1 Red + 3 Blue) and Gymnasium `NetForge/Blue-v4`. Blue trains on SIEM belief graphs, not the true map.
- 📟 **20** — `netforge` CLI: `run`, `evaluate`, `questions`, HTML replay. Packs for hospital, cloud, APT, IoT, OT.
- 🏗️ **20** — Arena train / dev / hidden splits.

🏷️ [Changelog](docs/changelog.md) · 📘 [Docs](https://reforcemind.github.io/NetForge_RL/)

***

## The problem

The job is a SOC shift, not a capture-the-flag. Blue only ever sees a log
buffer: Sysmon-like events that can arrive late, drop, or never fire if Red
stays quiet. Isolation is slow and takes hosts off the mission. A policy that
reads the simulator's compromise map, or that wins by unplugging the plant, is
not a defender.

That is what most cyber RL gyms still train. Blue gets something close to true
host state. Actions finish in one step. The paper reports mean reward against
one scripted Red. The agent looks strong in that setup and fails as soon as the
logs are incomplete or the attacker changes.

NetForge is a gym for the first job: train and rank on SIEM, SLA, and a Red
population. Trainers stay in your repo.

## Compared to other gyms

CybORG / CAGE is the competition stack papers already cite. CyberBattleSim and
NASim are graph capture and pentest. Yawning Titan is abstract graph defense.
NetForge is the SOC gym: delayed logs, durative actions, PettingZoo ids, Arena
metrics.

| | Blue observation | Time | What you rank | API |
|---|---|---|---|---|
| [CybORG](https://github.com/cage-challenge/CybORG) / [CAGE](https://github.com/cage-challenge) | Host table, often close to true state | Mixed; many actions resolve in-step | Historically mean return vs scripted Red (B-line, Meander) | Custom env + wrappers |
| [CyberBattleSim](https://github.com/microsoft/CyberBattleSim) | Discovered attack graph | Instant node/credential actions | Red capture; Blue is thin | Gymnasium |
| [NASim](https://github.com/Jjschwartz/NetworkAttackSimulator) | Scan-revealed network | Instant exploits | Attacker success | Gymnasium |
| [Yawning Titan](https://github.com/dstl/YAWNING-TITAN) | Abstract graph nodes | Instant | Graph defense | Gymnasium |
| **NetForge** | SIEM buffer + belief graph. Oracle is diagnostic only | Durative: exploits and isolates take ticks | Arena: mission, SLA, FPs, security, CVaR vs a Red population | PettingZoo + Gymnasium |

NetForge does not try to beat CybORG on host-type count. LICENSE still carries
CybORG / DSTG notices. The bet is the observation and the score: Blue trains on
logs, isolate-all wrecks SLA, and one campaign is not an eval.

Ranges / Caldera / FARLAND are emulation or adversary tooling. Use them when you
need packets on a wire. This repo is the pip-installable gym.

<p align="center">
  <img src="docs/_static/figures/why-this.svg" width="920" alt="SOC view vs god-mode, delayed logs, SLA vs isolate-all">
</p>

## What NetForge does

Each tick, agents pick `[action_type, host_index]`. Actions run, conflicts
resolve, SIEM fires, Blue obs update from the buffer.

**Red** — discover, exploit CVEs, escalate, exfil, or hit a PLC.
**Blue** — isolate, ACL, restore, decoys (`169.254.x.x`). Quiet Red may never
show up in logs.
**Kinetic** — `OverloadPLC` can end the episode for Blue.

```python
import netforge_rl
import gymnasium as gym
from pettingzoo import make

env = make('parallel', 'netforge/ransomware-v4', max_ticks=80)
obs, infos = env.reset(seed=0)
blue = gym.make('NetForge/Blue-v4', max_ticks=80)
```

<p align="center">
  <img src="docs/_static/figures/workflow.svg" width="920" alt="reset, observe SIEM, act, next tick">
</p>

## First run

Not on PyPI. Python 3.12+.

```bash
python -m pip install 'netforge-rl @ git+https://github.com/reforcemind/NetForge_RL'
netforge run hospital_ransomware --replay replay.html
```

From a clone:

```bash
git clone https://github.com/reforcemind/NetForge_RL.git
cd NetForge_RL
pip install -e ".[dev]"
netforge run ransomware --seed 0 --max-ticks 40
```

```bash
python -m netforge --help
```

| Piece | |
|---|---|
| Agents | `red_operator`, `blue_dmz`, `blue_internal`, `blue_restricted` |
| Action | `MultiDiscrete([32, 100])`, mask `int8[132]` |
| Blue obs | vector + SIEM embedding + belief graph |
| Families | ransomware, APT, cloud, IoT, OT/Stuxnet |
| Trainers | your code. Not bundled. |

Gate: `scripts/verify_all.sh` or `scripts/verify_local.ps1`. GitHub Actions (`ci.yml`) is ruff + pytest + Sphinx, and deploys GitHub Pages on `main`.

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

MIT, with CybORG / DSTG notices in [LICENSE](LICENSE).
