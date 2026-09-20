# Getting Started

Python 3.12+. Install from git (not PyPI).

```bash
python -m pip install 'netforge-rl @ git+https://github.com/reforcemind/NetForge_RL'
netforge run hospital_ransomware --replay replay.html
```

That spins Red vs Blue on a hospital ransomware pack and writes an HTML replay
of hosts, SIEM, and rewards.

```bash
git clone https://github.com/reforcemind/NetForge_RL.git
cd NetForge_RL
pip install -e ".[dev]"
```

```python
import netforge_rl
from pettingzoo import make
from netforge_rl.environment.graph_wrapper import GraphObservationWrapper

env = GraphObservationWrapper(make("parallel", "netforge/ransomware-v4", max_ticks=80))
obs, infos = env.reset(seed=0)
```

`infos["blue_dmz"]["graph"]` is what Blue believes from SIEM.
One Blue vs a scripted attacker: `gym.make("NetForge/Blue-v4")`.

| | |
|---|---|
| Env ids | [Python](python-quickstart) |
| Red / Blue / SIEM | [Cyber](cybersec/overview) |
| Packs | [Scenarios](scenarios) |
| Your policy | [Submissions](submissions) |
| Gate | [Verification](guides/verification) |
