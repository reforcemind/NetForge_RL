# Custom Scenarios

NetForge allows researchers to define custom scenarios by subclassing `BaseScenario`. Custom scenarios define agent roles, custom reward functions, initial network states, and episode termination criteria.

## Scenario Implementation

```python
"""Custom scenario implementation."""
from netforge_rl.scenarios.base_scenario import BaseScenario


class CustomEnterpriseScenario(BaseScenario):
    """Custom scenario modeling enterprise lateral movement."""

    def calculate_reward(self, agent_id, global_state, effect=None):
        """Calculate step reward for the given agent."""
        reward = 0.0

        # Penalize action costs (e.g. bandwidth or computation)
        if effect and getattr(effect, "cost", 0) > 0:
            reward -= effect.cost * 0.05

        if "red" in agent_id:
            return self._calculate_red_reward(global_state, effect, reward)
        return self._calculate_blue_reward(global_state, effect, reward)

    def _calculate_red_reward(self, global_state, effect, base_reward):
        if not (effect and effect.success and effect.state_deltas):
            return base_reward

        reward = base_reward
        for key, val in effect.state_deltas.items():
            if "privilege" in key and val == "Root":
                reward += 5.0
            if "system_integrity" in key and val == "kinetic_destruction":
                reward += 500.0
        return reward

    def _calculate_blue_reward(self, global_state, effect, base_reward):
        hosts = global_state.all_hosts.values()
        healthy_online = sum(
            1 for h in hosts if h.compromised_by == "None" and h.status != "isolated"
        )
        total_hosts = max(1, len(global_state.all_hosts))
        return base_reward + (healthy_online / total_hosts)

    def check_termination(self, global_state):
        """Determine episode termination status for each agent."""
        kinetic_failure = any(
            h.system_integrity == "kinetic_destruction"
            for h in global_state.all_hosts.values()
        )
        return {agent: kinetic_failure for agent in self.agents}
```

## Running Custom Scenarios

Load your custom scenario in Python:

```python
from netforge_rl.environment.parallel_env import NetForgeRLEnv

env = NetForgeRLEnv({"scenario_type": "custom_enterprise"})
```
