# Custom Scenarios

NetForge RL is highly extensible. The core mechanics (e.g., resolving an exploit against a vulnerability) are handled by the functional core, but you define the **win conditions** and **rewards** via Scenarios.

## The BaseScenario Interface

All custom scenarios must inherit from `netforge_rl.scenarios.base_scenario.BaseScenario`. You are required to implement two core methods:

1. `calculate_reward`: Calculates the scalar reward signal for a specific agent after a step.
2. `check_termination`: Determines if the episode has ended (Win/Loss).

## Example: Building a Ransomware Scenario

Here is an abbreviated example of how you can build a scenario where the Red team's goal is to compromise the Corporate subnet, and the Blue team's goal is to maintain uptime.

```python
from typing import Dict
from netforge_rl.scenarios.base_scenario import BaseScenario
from netforge_rl.core.state import GlobalNetworkState
from netforge_rl.core.action import ActionEffect

class CustomRansomware(BaseScenario):
    def __init__(self, agents):
        self.agents = agents

    def calculate_reward(
        self, agent_id: str, global_state: GlobalNetworkState, effect: ActionEffect = None
    ) -> float:
        reward = 0.0
        
        # Penalize agents for taking computationally expensive actions
        if effect and getattr(effect, 'cost', 0) > 0:
            reward -= effect.cost * 0.05

        if 'red' in agent_id:
            return self._red_reward(global_state, effect)
        else:
            return self._blue_reward(global_state, effect)

    def _red_reward(self, global_state, effect):
        reward = 0.0
        if effect and effect.success and effect.state_deltas:
            # Reward successful privilege escalation
            for delta_key, delta_val in effect.state_deltas.items():
                if 'privilege' in delta_key and delta_val == 'Root':
                    reward += 5.0
                if 'system_integrity' in delta_key and delta_val == 'kinetic_destruction':
                    reward += 10000.0  # Massive payout for critical impact
        return reward

    def _blue_reward(self, global_state, effect):
        # Reward proportional to healthy, unisolated hosts
        healthy = sum(
            1 for h in global_state.all_hosts.values()
            if h.compromised_by == 'None' and h.status != 'isolated'
        )
        return healthy / len(global_state.all_hosts)

    def check_termination(self, global_state: GlobalNetworkState) -> Dict[str, bool]:
        # End episode if any SCADA PLC is destroyed
        if any(h.system_integrity == 'kinetic_destruction' for h in global_state.all_hosts.values()):
            return {agent: True for agent in self.agents}
            
        return {agent: False for agent in self.agents}
```

## Integrating the Scenario

Once your scenario is built, you can pass its type name into the environment configuration dictionary if registered, or instantiate it directly and pass it into your custom loop.