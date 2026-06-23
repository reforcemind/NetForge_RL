from typing import Dict
from netforge_rl.scenarios.base_scenario import BaseScenario


class CloudHybridScenario(BaseScenario):
    """Cloud-style hybrid scenario."""

    name = 'cloud_hybrid'

    MAX_STEP_REWARD = 30.0
    SECURE_BREACH_REWARD = 25.0
    SECURE_LOSS_PENALTY = -30.0
    SLA_BONUS_SCALE = 1.0

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
                    if not host:
                        continue
                    if state.get_subnet_name(host.subnet_cidr) == 'Secure':
                        r += self.SECURE_BREACH_REWARD
                    elif state.get_subnet_name(host.subnet_cidr) == 'DMZ':
                        r += 1.0
                    else:
                        r += 2.0
        if effect and not effect.success:
            r -= 0.1
        return r

    def _blue(self, state, effect):
        r = 0.0
        secure_hosts = [
            h
            for h in state.all_hosts.values()
            if state.get_subnet_name(h.subnet_cidr) == 'Secure'
        ]
        if secure_hosts:
            secure_clean = sum(1 for h in secure_hosts if h.compromised_by == 'None')
            r += (secure_clean / len(secure_hosts)) * self.SLA_BONUS_SCALE
            if any(h.compromised_by != 'None' for h in secure_hosts):
                r += self.SECURE_LOSS_PENALTY

        if effect and effect.success and isinstance(effect.state_deltas, dict):
            for k, v in effect.state_deltas.items():
                if 'status' in k and v == 'isolated':
                    ip = k.split('/')[1] if '/' in k else None
                    host = state.all_hosts.get(ip) if ip else None
                    if host and state.get_subnet_name(host.subnet_cidr) == 'Secure':
                        r += 5.0
                    elif host and host.compromised_by != 'None':
                        r += 1.5
                    else:
                        r -= 1.0
        return r

    def check_termination(self, state) -> Dict[str, bool]:
        secure_hosts = [
            h
            for h in state.all_hosts.values()
            if state.get_subnet_name(h.subnet_cidr) == 'Secure'
        ]
        if secure_hosts and all(h.compromised_by != 'None' for h in secure_hosts):
            return {a: True for a in self.agents}
        return {a: False for a in self.agents}
