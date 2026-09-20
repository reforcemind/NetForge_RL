# Custom scenarios

Subclass `BaseScenario`: `calculate_reward(agent_id, global_state, effect)` and
`check_termination(global_state)`. Exploits stay in the functional core.

```python
from netforge_rl.scenarios.base_scenario import BaseScenario

class CustomRansomware(BaseScenario):
    def calculate_reward(self, agent_id, global_state, effect=None):
        reward = 0.0
        if effect and getattr(effect, 'cost', 0) > 0:
            reward -= effect.cost * 0.05
        if 'red' in agent_id:
            return self._red_reward(global_state, effect)
        return self._blue_reward(global_state, effect)

    def _red_reward(self, global_state, effect):
        if not (effect and effect.success and effect.state_deltas):
            return 0.0
        r = 0.0
        for key, val in effect.state_deltas.items():
            if 'privilege' in key and val == 'Root':
                r += 5.0
            if 'system_integrity' in key and val == 'kinetic_destruction':
                r += 10000.0
        return r

    def _blue_reward(self, global_state, effect):
        hosts = global_state.all_hosts.values()
        healthy = sum(
            1 for h in hosts
            if h.compromised_by == 'None' and h.status != 'isolated'
        )
        return healthy / len(global_state.all_hosts)

    def check_termination(self, global_state):
        dead = any(
            h.system_integrity == 'kinetic_destruction'
            for h in global_state.all_hosts.values()
        )
        return {agent: dead for agent in self.agents}
```

`ot_stuxnet`: PLC ODE physics, `OverloadPLC` moves setpoints, Blue has ~10 ticks
after `SCADA_PHYSICAL_ALARM` to isolate. Tables: {doc}`../architecture/ot_physics`.

```python
env = NetForgeRLEnv({'scenario_type': 'ot_stuxnet', 'max_ticks': 300})
env = NetForgeRLEnv({'scenario_type': 'custom_ransomware'})
```
