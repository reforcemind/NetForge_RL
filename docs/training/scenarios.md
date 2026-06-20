# Custom Scenarios

Custom scenarios define the **win conditions** and **reward tensors** for the environment. Core resolution logic (e.g., executing an exploit) is handled independently by the `Functional Core`.

## BaseScenario Interface

Scenarios must inherit from `netforge_rl.scenarios.base_scenario.BaseScenario` and implement two abstract methods:

1. `calculate_reward(self, agent_id, global_state, effect)`: Returns a scalar float representing the delta reward for the current step.
2. `check_termination(self, global_state)`: Returns a dictionary mapping agent IDs to boolean terminal states.

## Configuration: Ransomware

In a Ransomware configuration, the Red agent seeks to compromise system integrity flags, while the Blue agent seeks to maximize healthy host ratios.

### 1. Structure and Initialization

```python
from netforge_rl.scenarios.base_scenario import BaseScenario

class CustomRansomware(BaseScenario):
    def __init__(self, agents):
        self.agents = agents
```

### 2. Base Reward Routing

A central `calculate_reward` method routes scalar logic to specific agent functions, applying global penalties (like action cost) universally.

```python
    def calculate_reward(self, agent_id, global_state, effect=None):
        reward = 0.0
        
        # Base penalty for taking actions
        if effect and getattr(effect, 'cost', 0) > 0:
            reward -= effect.cost * 0.05

        if 'red' in agent_id:
            return self._red_reward(global_state, effect)
        return self._blue_reward(global_state, effect)
```

### 3. Red Agent Logic

Red agents are rewarded exclusively for successfully modifying the `GlobalNetworkState` access and integrity tensors.

```python
    def _red_reward(self, global_state, effect):
        reward = 0.0
        if effect and effect.success and effect.state_deltas:
            for key, val in effect.state_deltas.items():
                if 'privilege' in key and val == 'Root':
                    reward += 5.0
                if 'system_integrity' in key and val == 'kinetic_destruction':
                    reward += 10000.0
        return reward
```

### 4. Blue Agent Logic

Blue agents are rewarded proportionally to the number of nodes maintaining a baseline (healthy) state vector.

```python
    def _blue_reward(self, global_state, effect):
        healthy = sum(
            1 for h in global_state.all_hosts.values()
            if h.compromised_by == 'None' and h.status != 'isolated'
        )
        return healthy / len(global_state.all_hosts)
```

### 5. Termination Logic

The episode halts immediately if a critical Cyber-Physical node transitions to a destroyed state.

```python
    def check_termination(self, global_state):
        if any(h.system_integrity == 'kinetic_destruction' for h in global_state.all_hosts.values()):
            return {agent: True for agent in self.agents}
            
        return {agent: False for agent in self.agents}
```

## Registering Configurations

Scenario classes are registered dynamically via the `parallel_env` instantiation dictionary:

```python
env = NetForgeRLEnv({'scenario_type': 'custom_ransomware'})
```