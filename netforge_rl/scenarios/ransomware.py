from netforge_rl.scenarios.base_scenario import PADDING_SUBNET, BaseScenario


class RansomwareScenario(BaseScenario):
    """Ransomware scenario."""

    MAX_STEP_REWARD = 5.0

    REWARD_WEIGHTS = {
        'red_weights': {
            'privilege_user': 3.0,
            'privilege_root': 5.0,
            'system_compromised': 10.0,
            'kinetic_destruction': 10000.0,
            'host_owned': 2.0,
            'host_discovered': 0.5,
            'scan_result': 0.3,
            'intel_shared': 1.0,
            'hit_decoy': -3.0,
            'kernel_panic': -5.0,
            'action_cost': -0.05,
            'failed_action': -0.1,
        },
        'blue_weights': {
            'correct_isolation': 5.0,
            'false_positive_isolation': -2.0,
            'host_cleaned': 3.0,
            'service_restored': 1.0,
            'decoy_deployed': 1.0,
            'kinetic_failure': -10000.0,
            'health_ratio': 1.0,
            'compromised_ratio': -2.0,
            'business_downtime': -5.0,
            'action_cost': -0.05,
            'failed_action': -0.1,
        },
    }

    def _red_reward(self, agent_id, state, effect):
        r = 0.0
        for attr, _ip, val in self._iter_deltas(effect):
            if attr == 'privilege':
                r += 3.0 if val == 'User' else 5.0
            elif attr == 'system_integrity':
                if val == 'compromised':
                    r += 10.0
                elif val == 'kinetic_destruction':
                    r += 10000.0
            elif attr == 'compromised_by' and val != 'None':
                r += 2.0

        if effect and effect.observation_data:
            obs = effect.observation_data
            if 'discovered_hosts' in obs:
                discovered = obs['discovered_hosts']
                count = len(discovered) if isinstance(discovered, (list, set)) else 1
                r += count * 0.5
            if 'scan_results' in obs:
                r += 0.3
            if 'shared' in obs:
                r += 1.0
            if 'Failed against Decoy' in str(obs.values()):
                r -= 3.0
            elif 'kernel panic' in str(obs.values()):
                r -= 5.0
        return r + self._failed_penalty(effect)

    def _blue_reward(self, agent_id, state, effect):
        r = 0.0
        for attr, ip, val in self._iter_deltas(effect):
            if attr == 'status' and val == 'isolated':
                host = state.all_hosts.get(ip)
                if host and host.compromised_by != 'None':
                    r += 5.0  # correctly quarantined a compromised host
                else:
                    r -= 2.0  # false positive — isolated a clean host
            elif attr == 'privilege' and val == 'None':
                r += 3.0  # cleaned a host
            elif attr == 'status' and val == 'online':
                r += 1.0  # restored connectivity
            elif attr == 'decoy' and val != 'inactive':
                r += 1.0  # proactive defense
            elif attr == 'system_integrity' and val == 'kinetic_destruction':
                r -= 10000.0  # failed to prevent kinetic destruction

        total = max(len(state.all_hosts), 1)
        healthy = sum(
            1
            for h in state.all_hosts.values()
            if h.compromised_by == 'None' and h.status != 'isolated'
        )
        r += (healthy / total) * 1.0

        compromised = sum(
            1
            for h in state.all_hosts.values()
            if h.compromised_by != 'None' and h.status != 'isolated'
        )
        if compromised > 0:
            r -= (compromised / total) * 2.0

        isolated = sum(
            1
            for h in state.all_hosts.values()
            if h.status == 'isolated' and h.subnet_cidr != PADDING_SUBNET
        )
        if isolated > 0:
            business_loss = (isolated / total) * 5.0
            state.business_downtime_score += business_loss
            r -= business_loss
        return r + self._failed_penalty(effect)

    def _is_terminal(self, state):
        targets = [
            h
            for h in state.all_hosts.values()
            if state.get_subnet_name(h.subnet_cidr) in ('Corporate', 'Secure')
        ]
        if targets and all(
            getattr(h, 'system_integrity', 'clean') == 'compromised' for h in targets
        ):
            return True
        return self._any_kinetic(state)
