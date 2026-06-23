from __future__ import annotations
from typing import Dict
from netforge_rl.scenarios.base_scenario import BaseScenario


class OTStuxnetScenario(BaseScenario):
    name = 'ot_stuxnet'

    KINETIC_REWARD = 10_000.0
    PER_TIER_BONUS = 4.0  # Red gains for each new tier they own
    BLUE_HOLD_BONUS = 0.5  # per step while no OT host is compromised
    BLUE_KINETIC_PENALTY = -10_000.0

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
                    ip = k.split('/')[1] if '/' in k else None
                    host = state.all_hosts.get(ip) if ip else None
                    if host:
                        if state.get_subnet_name(host.subnet_cidr) == 'OT':
                            r += self.PER_TIER_BONUS * 3
                        else:
                            r += self.PER_TIER_BONUS
        if any(
            getattr(h, 'system_integrity', 'clean') == 'kinetic_destruction'
            for h in state.all_hosts.values()
        ):
            r += self.KINETIC_REWARD
        if effect and not effect.success:
            r -= 0.1
        return r

    def _blue(self, state, effect):
        r = 0.0
        # Bonus per step where the OT tier is still safe.
        ot_clean = all(
            h.compromised_by == 'None'
            for h in state.all_hosts.values()
            if state.get_subnet_name(h.subnet_cidr) == 'OT'
        )
        if ot_clean:
            r += self.BLUE_HOLD_BONUS

        # Catastrophic penalty if any PLC takes kinetic damage.
        if any(
            getattr(h, 'system_integrity', 'clean') == 'kinetic_destruction'
            for h in state.all_hosts.values()
        ):
            r += self.BLUE_KINETIC_PENALTY

        if effect and effect.success and isinstance(effect.state_deltas, dict):
            for k, v in effect.state_deltas.items():
                if 'status' in k and v == 'isolated':
                    ip = k.split('/')[1] if '/' in k else None
                    host = state.all_hosts.get(ip) if ip else None
                    if host and state.get_subnet_name(host.subnet_cidr) == 'OT':
                        r += 6.0  # Isolating an OT host is high-value defense
        return r

    def check_termination(self, state) -> Dict[str, bool]:
        if any(
            getattr(h, 'system_integrity', 'clean') == 'kinetic_destruction'
            for h in state.all_hosts.values()
        ):
            return {a: True for a in self.agents}
        return {a: False for a in self.agents}
