# PettingZoo and Gymnasium

```python
import netforge_rl
import gymnasium as gym
from pettingzoo import make

env = make("parallel", "netforge/ransomware-v4", max_ticks=200)
single = gym.make("NetForge/Blue-v4", max_ticks=200)
```

| API | Id |
|---|---|
| PettingZoo parallel | `netforge/ransomware-v4`, `netforge/arena-v4` |
| PettingZoo AEC | same ids |
| Gymnasium | `NetForge/Blue-v4`, alias `NetForge-v4` |

`gym.make("netforge_rl:NetForge/Blue-v4")` also works. Mask is flat `int8[132]`;
`CyberActionSpace.sample(mask=…)` accepts it.

Farama listing (docs PR to PettingZoo `third_party_envs`): Parallel + AEC
`netforge/ransomware-v4`, Gymnasium `NetForge/Blue-v4`,
https://reforcemind.github.io/NetForge_RL/
