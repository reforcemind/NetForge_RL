# NetForge

<p class="fe-lede">Cybersecurity gym for reinforcement learning. Red compromises hosts. Blue contains them from SIEM, not from an oracle. Train with PettingZoo or Gymnasium.</p>

```{raw} html
<ul class="fe-chips">
  <li>Red vs Blue</li>
  <li>SIEM telemetry</li>
  <li>PettingZoo</li>
  <li>Gymnasium</li>
</ul>

<div class="fe-metrics">
  <div class="fe-metric">
    <span class="fe-metric-value">1 Red · 3 Blue</span>
    <span class="fe-metric-label">DMZ, internal, restricted</span>
  </div>
  <div class="fe-metric">
    <span class="fe-metric-value">SIEM only</span>
    <span class="fe-metric-label">Blue never sees the true map</span>
  </div>
  <div class="fe-metric">
    <span class="fe-metric-value">durative actions</span>
    <span class="fe-metric-label">exploits and isolates take ticks</span>
  </div>
</div>

<nav class="fe-jump" aria-label="Start here">
  <a class="fe-jump-card" href="getting-started.html">
    <span class="fe-jump-kicker">01</span>
    <strong>Start</strong>
    <span>Install, run a ransomware pack.</span>
  </a>
  <a class="fe-jump-card" href="python-quickstart.html">
    <span class="fe-jump-kicker">02</span>
    <strong>Python</strong>
    <span><code>gym.make</code> / <code>pettingzoo.make</code></span>
  </a>
  <a class="fe-jump-card" href="cybersec/overview.html">
    <span class="fe-jump-kicker">03</span>
    <strong>Cyber</strong>
    <span>Red, Blue, SIEM, OT</span>
  </a>
  <a class="fe-jump-card" href="scenarios.html">
    <span class="fe-jump-kicker">04</span>
    <strong>Scenarios</strong>
    <span>Hospital, cloud, plant, IoT</span>
  </a>
  <a class="fe-jump-card" href="architecture/overview.html">
    <span class="fe-jump-kicker">05</span>
    <strong>Architecture</strong>
    <span>Tick loop, logs, graphs</span>
  </a>
  <a class="fe-jump-card" href="api/python.html">
    <span class="fe-jump-kicker">06</span>
    <strong>API</strong>
    <span>Env, SIEM, NLP</span>
  </a>
</nav>
```

## News

**September 2026**

- Gym ids: PettingZoo `netforge/ransomware-v4`, Gymnasium `NetForge/Blue-v4`.
- `netforge` CLI: `run`, `evaluate`, `questions`, HTML replay.
- Arena train / dev / hidden. Typed `EnvConfig`. {doc}`changelog`

## The problem

The job is a SOC shift, not a capture-the-flag. Blue only ever sees a log
buffer: Sysmon-like events that can arrive late, drop, or never fire if Red
stays quiet. Isolation is slow and takes hosts off the mission. A policy that
reads the simulator's compromise map, or that wins by unplugging the plant, is
not a defender.

Most cyber RL gyms still train a different job. Blue gets something close to
true host state. Actions finish in one step. The paper reports mean reward
against one scripted Red. The agent looks strong there and fails as soon as the
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

Ranges, Caldera, and FARLAND are emulation or adversary tooling. Use them when
you need packets on a wire. This repo is the pip-installable gym.

## What NetForge does

Red starts blind, discovers hosts, exploits CVEs, escalates, exfiltrates or
hits a PLC. Blue isolates, restores, drops decoys, and reads Sysmon-like logs
that can be late or noisy. Padding hosts (`169.254.x.x`) are decoys.

```{image} _static/figures/purpose.svg
:alt: Red, Blue, and the network
:class: fe-fig
```

```{image} _static/figures/workflow.svg
:alt: reset, observe SIEM, act, next tick
:class: fe-fig
```

```{image} _static/figures/why-this.svg
:alt: God-mode vs SIEM, instant logs vs delay, isolate-all vs SLA
:class: fe-fig
```

## Install

```bash
python -m pip install 'netforge-rl @ git+https://github.com/reforcemind/NetForge_RL'
netforge run hospital_ransomware --replay replay.html
```

```python
import netforge_rl
import gymnasium as gym
from pettingzoo import make

env = make('parallel', 'netforge/ransomware-v4', max_ticks=80)
obs, infos = env.reset(seed=0)
blue = gym.make('NetForge/Blue-v4', max_ticks=80)
```

Not on PyPI. Python 3.12+. Trainers stay in your repo.

MIT, with CybORG / DSTG notices in the LICENSE file.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Start
Overview <self>
getting-started
python-quickstart
arena
questions
submissions
capabilities
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Guides
scenarios
baselines
datasets
backends
diagnostics/overview
environment/difficulty
environment/reproducibility
environment/graph_observations
training/curriculum
training/rllib
training/scenarios
guides/verification
interop/pettingzoo
interop/soc_export
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Benchmarks
benchmarks/overview
benchmarks/baselines
benchmarks/self_play
benchmarks/run
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Architecture
architecture/overview
architecture/dynamic_topology
architecture/ot_physics
architecture/multimodal_obs
architecture/zero_trust
architecture/sim2real
architecture/nlp_siem
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Cyber
cybersec/overview
cybersec/threat_model
cybersec/red_actions
cybersec/blue_actions
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: API
api/python
api/environment
api/siem
api/nlp
api/sim2real
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Contribute
contributing
research
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project
roadmap
product-direction
DATASHEET
changelog
```
