from netforge_rl.scenarios.base_scenario import BaseScenario


class CloudHybridScenario(BaseScenario):
    """Cloud-style hybrid scenario."""

    name = 'cloud_hybrid'

    MAX_STEP_REWARD = 30.0
    SECURE_BREACH_REWARD = 25.0
    SECURE_LOSS_PENALTY = -30.0
    SLA_BONUS_SCALE = 1.0

    REWARD_WEIGHTS = {
        'red_weights': {
            'secure_breach': SECURE_BREACH_REWARD,
            'dmz_breach': 1.0,
            'internal_breach': 2.0,
            'failed_action': -0.1,
            'action_cost': -0.05,
        },
        'blue_weights': {
            'secure_sla_scale': SLA_BONUS_SCALE,
            'secure_loss': SECURE_LOSS_PENALTY,
            'secure_isolation': 5.0,
            'compromised_isolation': 1.5,
            'false_positive_isolation': -1.0,
            'action_cost': -0.05,
        },
    }

    def _red_reward(self, agent_id, state, effect):
        r = 0.0
        for attr, ip, v in self._iter_deltas(effect):
            if attr == 'privilege' and v in ('User', 'Root'):
                host = state.all_hosts.get(ip)
                if not host:
                    continue
                subnet = state.get_subnet_name(host.subnet_cidr)
                if subnet == 'Secure':
                    r += self.SECURE_BREACH_REWARD
                elif subnet == 'DMZ':
                    r += 1.0
                else:
                    r += 2.0
        return r + self._failed_penalty(effect)

    def _blue_reward(self, agent_id, state, effect):
        r = 0.0
        secure = self._hosts_in_subnet(state, 'Secure')
        if secure:
            clean = sum(1 for h in secure if h.compromised_by == 'None')
            r += (clean / len(secure)) * self.SLA_BONUS_SCALE
            if any(h.compromised_by != 'None' for h in secure):
                r += self.SECURE_LOSS_PENALTY

        for attr, ip, v in self._iter_deltas(effect):
            if attr == 'status' and v == 'isolated':
                host = state.all_hosts.get(ip)
                if host and state.get_subnet_name(host.subnet_cidr) == 'Secure':
                    r += 5.0
                elif host and host.compromised_by != 'None':
                    r += 1.5
                else:
                    r -= 1.0
        return r

    def _is_terminal(self, state):
        secure = self._hosts_in_subnet(state, 'Secure')
        return bool(secure) and all(h.compromised_by != 'None' for h in secure)
