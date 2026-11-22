# NetForge

<p class="fe-lede">A reinforcement learning environment for multi-agent network defense and realistic SOC operations. Red compromises hosts under fog of war; Blue detects and contains threats through SIEM telemetry and belief-state graphs.</p>

```{raw} html
<ul class="fe-chips">
  <li>Red vs Blue</li>
  <li>SIEM Telemetry</li>
  <li>Durative Actions</li>
  <li>PettingZoo &amp; Gymnasium</li>
</ul>

<div class="fe-metrics">
  <div class="fe-metric">
    <span class="fe-metric-value">1 Red · 3 Blue</span>
    <span class="fe-metric-label">DMZ, internal, and restricted zones</span>
  </div>
  <div class="fe-metric">
    <span class="fe-metric-value">SIEM Telemetry</span>
    <span class="fe-metric-label">Blue observes event logs, not ground-truth state</span>
  </div>
  <div class="fe-metric">
    <span class="fe-metric-value">Durative Actions</span>
    <span class="fe-metric-label">Multi-tick actions with realistic latency</span>
  </div>
</div>

<nav class="fe-jump" aria-label="Documentation navigation">
  <a class="fe-jump-card" href="getting-started.html">
    <span class="fe-jump-kicker">01</span>
    <strong>Start</strong>
    <span>Install NetForge and run a simulation.</span>
  </a>
  <a class="fe-jump-card" href="python-quickstart.html">
    <span class="fe-jump-kicker">02</span>
    <strong>Python</strong>
    <span>Gymnasium and PettingZoo integration.</span>
  </a>
  <a class="fe-jump-card" href="cybersec/overview.html">
    <span class="fe-jump-kicker">03</span>
    <strong>Cyber</strong>
    <span>Threat model, Red tactics, Blue defenses.</span>
  </a>
  <a class="fe-jump-card" href="scenarios.html">
    <span class="fe-jump-kicker">04</span>
    <strong>Scenarios</strong>
    <span>Ransomware, APT, cloud, and OT packs.</span>
  </a>
  <a class="fe-jump-card" href="architecture/overview.html">
    <span class="fe-jump-kicker">05</span>
    <strong>Architecture</strong>
    <span>Tick resolution, action queues, and telemetry.</span>
  </a>
  <a class="fe-jump-card" href="arena.html">
    <span class="fe-jump-kicker">06</span>
    <strong>Evaluation</strong>
    <span>Arena splits, SLA metrics, and leaderboard.</span>
  </a>
</nav>
```

```{image} _static/figures/purpose.svg
:alt: NetForge Architecture
:class: fe-fig
```

## Overview

In enterprise security operations, defenders never see an omniscient map of compromised machines. Instead, they operate inside an asymmetric information environment:

- **SIEM Telemetry**: Defenders monitor event streams (process creation, authentication, network connections) that can be delayed, dropped, or absent if an adversary maintains stealth.
- **Durative Execution**: Network scans, exploits, and host isolations do not execute instantaneously. Actions run across ticks and can be preempted or cancelled by opposing moves.
- **Operational SLAs**: Isolation stops an attack but takes services offline. NetForge scores defenders on operational availability alongside threat containment.

## Environment Comparison

| Feature | CybORG / CAGE | CyberBattleSim | Yawning Titan | NetForge |
|---|---|---|---|---|
| **Blue Observation** | Host table (near-oracle) | Attack graph | Abstract graph nodes | **SIEM event buffer + belief graph** |
| **Action Resolution** | Instant / mixed | Instantaneous | Instantaneous | **Durative (multi-tick queue)** |
| **Defensive Penalty** | Reward penalty | N/A | Heuristic cost | **SLA uptime & false positive tracking** |
| **API Standards** | Custom wrapper | Gymnasium | Gymnasium | **PettingZoo (MARL) + Gymnasium** |
| **Adversary Eval** | Scripted Red (B-line/Meander) | Static graph | Scripted agents | **Adversarial populations & tournaments** |

## Quickstart

```python
import gymnasium as gym
import netforge_rl

# Single-agent Gymnasium defense against scripted adversary
env = gym.make("NetForge/Blue-v4", max_ticks=80)
obs, info = env.reset(seed=42)

# Multi-agent PettingZoo parallel environment
from pettingzoo import make
env = make("parallel", "netforge/ransomware-v4", max_ticks=80)
obs, infos = env.reset(seed=42)
```

Run from the command line:

```bash
# Run a scenario pack and generate an HTML replay
netforge run hospital_ransomware --replay replay.html

# Evaluate a submission policy
netforge evaluate examples/submissions/heuristic_blue.py --seeds 0 1 --max-ticks 40
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Getting Started
getting-started
python-quickstart
scenarios
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Cyber Mechanics
cybersec/overview
cybersec/threat_model
cybersec/red_actions
cybersec/blue_actions
environment/graph_observations
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
:caption: Evaluation & Benchmarks
arena
benchmarks/overview
benchmarks/baselines
benchmarks/run
benchmarks/self_play
questions
submissions
capabilities
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Guides & Training
interop/pettingzoo
interop/soc_export
training/curriculum
training/rllib
training/scenarios
datasets
backends
environment/difficulty
environment/reproducibility
diagnostics/overview
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Reference & Project
api/python
api/environment
api/siem
api/nlp
api/sim2real
guides/verification
contributing
DATASHEET
changelog
```
