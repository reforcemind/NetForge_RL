from netforge_rl.scenarios.base_scenario import PADDING_SUBNET, BaseScenario


class AptEspionageScenario(BaseScenario):
    """APT espionage scenario."""

    MAX_STEP_REWARD = 20.0
    ACTION_COST_FACTOR = 0.1

    REWARD_WEIGHTS = {
        'red_weights': {
            'recon': 2.0,
            'breach': 10.0,
            'exfiltration': 20.0,
            'intel_shared': 5.0,
            'hit_decoy': -15.0,
            'persistence_online': 5.0,
            'persistence_isolated': -20.0,
            'action_cost': -0.1,
        },
        'blue_weights': {
            'host_clean': 1.0,
            'host_infected': -5.0,
            'infected_isolated': 20.0,
            'action_cost': -0.1,
        },
    }

    def _red_reward(self, agent_id, state, effect):
        r = 0.0
        if effect and effect.success:
            name = type(getattr(effect, 'action', None)).__name__
            obs = effect.observation_data
            if 'Discover' in name or 'Scan' in name:
                r += 2.0
            if 'Exploit' in name or 'Pivot' in name:
                r += 10.0
            if 'Exfiltrate' in name or 'Dump' in name or 'Impact' in name:
                r += 20.0
            if 'shared' in str(obs):
                r += 5.0
            if 'Failed against Decoy' in str(obs.values()):
                r -= 15.0

        for h in state.all_hosts.values():
            if h.subnet_cidr == PADDING_SUBNET:
                continue
            if h.privilege in ('User', 'Root'):
                if h.status == 'online':
                    r += 5.0
                elif h.status == 'isolated':
                    r -= 20.0
        return r

    def _blue_reward(self, agent_id, state, effect):
        r = 0.0
        for h in state.all_hosts.values():
            if h.subnet_cidr == PADDING_SUBNET:
                continue
            infected = h.privilege in ('User', 'Root')
            isolated = h.status == 'isolated'
            if not infected and not isolated:
                r += 1.0
            elif infected and not isolated:
                r -= 5.0
            elif infected and isolated:
                r += 20.0
        return r

    def _is_terminal(self, state):
        infected = [
            h for h in state.all_hosts.values() if h.privilege in ('User', 'Root')
        ]
        return bool(infected) and all(h.status == 'isolated' for h in infected)
