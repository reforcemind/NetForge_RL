from netforge_rl.scenarios.base_scenario import BaseScenario


class OTStuxnetScenario(BaseScenario):
    name = 'ot_stuxnet'

    MAX_STEP_REWARD = 10_000.0
    KINETIC_REWARD = 10_000.0
    PER_TIER_BONUS = 4.0
    BLUE_HOLD_BONUS = 0.5
    BLUE_KINETIC_PENALTY = -10_000.0

    REWARD_WEIGHTS = {
        'red_weights': {
            'ot_privilege': PER_TIER_BONUS * 3,
            'it_privilege': PER_TIER_BONUS,
            'kinetic_destruction': KINETIC_REWARD,
            'failed_action': -0.1,
            'action_cost': -0.05,
        },
        'blue_weights': {
            'ot_hold_bonus': BLUE_HOLD_BONUS,
            'kinetic_penalty': BLUE_KINETIC_PENALTY,
            'ot_isolation': 6.0,
            'action_cost': -0.05,
        },
    }

    def _red_reward(self, agent_id, state, effect):
        r = 0.0
        for attr, ip, v in self._iter_deltas(effect):
            if attr == 'privilege' and v in ('User', 'Root'):
                host = state.all_hosts.get(ip)
                if host:
                    if state.get_subnet_name(host.subnet_cidr) == 'OT_Subnet':
                        r += self.PER_TIER_BONUS * 3
                    else:
                        r += self.PER_TIER_BONUS
        if self._any_kinetic(state):
            r += self.KINETIC_REWARD
        return r + self._failed_penalty(effect)

    def _blue_reward(self, agent_id, state, effect):
        r = 0.0
        ot_hosts = self._hosts_in_subnet(state, 'OT_Subnet')
        if all(h.compromised_by == 'None' for h in ot_hosts):
            r += self.BLUE_HOLD_BONUS  # OT tier still safe this step
        if self._any_kinetic(state):
            r += self.BLUE_KINETIC_PENALTY  # catastrophic PLC damage

        for attr, ip, v in self._iter_deltas(effect):
            if attr == 'status' and v == 'isolated':
                host = state.all_hosts.get(ip)
                if host and state.get_subnet_name(host.subnet_cidr) == 'OT_Subnet':
                    r += 6.0  # isolating an OT host is high-value defense
        return r

    def _is_terminal(self, state):
        return self._any_kinetic(state)
