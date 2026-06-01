"""IoT Grid scenario — many low-value devices + a handful of high-value
controllers. Red is rewarded for bricking hosts; Blue is rewarded for
keeping the grid online.

Reward shape mirrors the ransomware scenario: per-step health bonus for
Blue, one-time compromise bonus for Red, large penalty if any controller
(domain controller, in the topology generator's sense) falls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from netforge_rl.scenarios.base_scenario import BaseScenario

if TYPE_CHECKING:
    from netforge_rl.core.action import ActionEffect
    from netforge_rl.core.state import GlobalNetworkState


class IoTGridScenario(BaseScenario):
    name = 'iot_grid'

    CONTROLLER_LOSS_PENALTY = -250.0
    HEALTH_BONUS_SCALE = 0.5

    def __init__(self, agents):
        self.agents = agents

    def calculate_reward(self, agent_id, global_state, effect=None):
        reward = 0.0
        if effect and getattr(effect, 'cost', 0) > 0:
            reward -= effect.cost * 0.05

        if 'red' in agent_id.lower():
            reward += self._red(global_state, effect)
        else:
            reward += self._blue(global_state, effect)
        return reward

    def _red(self, state, effect):
        r = 0.0
        if effect and effect.success and isinstance(effect.state_deltas, dict):
            for k, v in effect.state_deltas.items():
                if 'privilege' in k and v in ('User', 'Root'):
                    r += 2.0
                    ip = k.split('/')[1] if '/' in k else None
                    if ip and state.all_hosts.get(ip) and state.all_hosts[ip].is_domain_controller:
                        r += 40.0
                if 'status' in k and v == 'kernel_panic':
                    r += 5.0
        if effect and not effect.success:
            r -= 0.1
        return r

    def _blue(self, state, effect):
        r = 0.0
        total = max(len(state.all_hosts), 1)
        healthy = sum(
            1 for h in state.all_hosts.values()
            if h.compromised_by == 'None' and h.status == 'online'
        )
        r += (healthy / total) * self.HEALTH_BONUS_SCALE

        # Big penalty if any controller is compromised — load-bearing for grid.
        if any(
            h.is_domain_controller and h.compromised_by != 'None'
            for h in state.all_hosts.values()
        ):
            r += self.CONTROLLER_LOSS_PENALTY

        if effect and effect.success and isinstance(effect.state_deltas, dict):
            for k, v in effect.state_deltas.items():
                if 'status' in k and v == 'isolated':
                    ip = k.split('/')[1] if '/' in k else None
                    if ip and state.all_hosts.get(ip) and state.all_hosts[ip].compromised_by != 'None':
                        r += 3.0
                    else:
                        r -= 1.0
        return r

    def check_termination(self, state) -> Dict[str, bool]:
        # End the episode if every controller is lost.
        controllers = [h for h in state.all_hosts.values() if h.is_domain_controller]
        if controllers and all(h.compromised_by != 'None' for h in controllers):
            return {a: True for a in self.agents}
        return {a: False for a in self.agents}
