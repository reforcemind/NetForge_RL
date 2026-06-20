# Custom Scenarios

## BaseScenario Interface

Custom scenarios must inherit from `netforge_rl.scenarios.base_scenario.BaseScenario` and implement the following methods:

1. `calculate_reward`: Returns a scalar reward float per agent after a step.
2. `check_termination`: Returns a boolean dictionary indicating episode completion.

## Implementation Example

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
        
        if effect and getattr(effect, 'cost', 0) > 0:
            reward -= effect.cost * 0.05

        if 'red' in agent_id:
            return self._red_reward(global_state, effect)
        else:
            return self._blue_reward(global_state, effect)

    def _red_reward(self, global_state, effect):
        reward = 0.0
        if effect and effect.success and effect.state_deltas:
            for delta_key, delta_val in effect.state_deltas.items():
                if 'privilege' in delta_key and delta_val == 'Root':
                    reward += 5.0
                if 'system_integrity' in delta_key and delta_val == 'kinetic_destruction':
                    reward += 10000.0
        return reward

    def _blue_reward(self, global_state, effect):
        healthy = sum(
            1 for h in global_state.all_hosts.values()
            if h.compromised_by == 'None' and h.status != 'isolated'
        )
        return healthy / len(global_state.all_hosts)

    def check_termination(self, global_state: GlobalNetworkState) -> Dict[str, bool]:
        if any(h.system_integrity == 'kinetic_destruction' for h in global_state.all_hosts.values()):
            return {agent: True for agent in self.agents}
            
        return {agent: False for agent in self.agents}
```