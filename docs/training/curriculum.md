# Curriculum Learning

Network defense presents a difficult exploration problem for reinforcement learning policies. NetForge provides automated curriculum learning through `CurriculumWrapper`, gradually increasing environmental complexity as agents demonstrate competence.

## Curriculum Progression Phases

| Phase | Network Scale | Scenario Scope | Environmental Dynamics | Reward Multiplier |
|---|---|---|---|---|
| **Novice** | 5 hosts | Basic ransomware | Static network, fixed IPs, no background noise | $3.0\times$ |
| **Intermediate** | 25 hosts | Ransomware + Enterprise APT | Periodic DHCP IP reassignments every 80 ticks | $1.5\times$ |
| **Expert** | 100 hosts | Full scenario catalog | Dynamic topology churn, roaming subnets, high noise | $1.0\times$ |

## Usage

Wrap the base environment with `CurriculumWrapper`:

```python
from pettingzoo import make
from netforge_rl.environment.curriculum import CurriculumWrapper

base_env = make("parallel", "netforge/ransomware-v4")
env = CurriculumWrapper(
    base_env,
    advance_threshold=0.65,  # Advance phase when moving-average reward clears 0.65
    window_size=50,          # Moving average over 50 episodes
)

obs, infos = env.reset(seed=42)
```

The current curriculum phase and performance metrics are tracked under `info["__curriculum__"]`.
