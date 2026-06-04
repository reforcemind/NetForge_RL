from typing import Dict

from netforge_rl.scenarios.base_scenario import BaseScenario


_SECURE_CIDR = '10.0.1.0/24'
_DMZ_CIDR = '192.168.1.0/24'


class CloudHybridScenario(BaseScenario):
    """Cloud-style hybrid: Red wins by breaching the Secure subnet (ZTNA-gated);
    Blue wins by keeping SLA uptime + identity rotation hygiene.

    Red bonuses scale with how deep into the network the breach went; Blue gets
    a per-step bonus for every Secure host still clean and a one-time hit
    whenever a Secure host falls.
    """

    name = 'cloud_hybrid'

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
                    if host.subnet_cidr == _SECURE_CIDR:
                        r += self.SECURE_BREACH_REWARD
                    elif host.subnet_cidr == _DMZ_CIDR:
                        r += 1.0
                    else:
                        r += 2.0
        if effect and not effect.success:
            r -= 0.1
        return r

    def _blue(self, state, effect):
        r = 0.0
        secure_hosts = [
            h for h in state.all_hosts.values() if h.subnet_cidr == _SECURE_CIDR
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
                    if host and host.subnet_cidr == _SECURE_CIDR:
                        r += 5.0
                    elif host and host.compromised_by != 'None':
                        r += 1.5
                    else:
                        r -= 1.0
        return r

    def check_termination(self, state) -> Dict[str, bool]:
        secure_hosts = [
            h for h in state.all_hosts.values() if h.subnet_cidr == _SECURE_CIDR
        ]
        if secure_hosts and all(h.compromised_by != 'None' for h in secure_hosts):
            return {a: True for a in self.agents}
        return {a: False for a in self.agents}
