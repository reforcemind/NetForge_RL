from netforge_rl.scenarios.base_scenario import BaseScenario


class IoTGridScenario(BaseScenario):
    name = 'iot_grid'

    MAX_STEP_REWARD = 250.0
    CONTROLLER_LOSS_PENALTY = -250.0
    HEALTH_BONUS_SCALE = 0.5

    REWARD_WEIGHTS = {
        'red_weights': {
            'privilege': 2.0,
            'controller_breach': 40.0,
            'kernel_panic': 5.0,
            'failed_action': -0.1,
            'action_cost': -0.05,
        },
        'blue_weights': {
            'health_ratio_scale': HEALTH_BONUS_SCALE,
            'controller_loss': CONTROLLER_LOSS_PENALTY,
            'correct_isolation': 3.0,
            'false_positive_isolation': -1.0,
            'action_cost': -0.05,
        },
    }

    def _red_reward(self, agent_id, state, effect):
        r = 0.0
        for attr, ip, v in self._iter_deltas(effect):
            if attr == 'privilege' and v in ('User', 'Root'):
                r += 2.0
                host = state.all_hosts.get(ip)
                if host and host.is_domain_controller:
                    r += 40.0
            if attr == 'status' and v == 'kernel_panic':
                r += 5.0
        return r + self._failed_penalty(effect)

    def _blue_reward(self, agent_id, state, effect):
        total = max(len(state.all_hosts), 1)
        healthy = sum(
            1
            for h in state.all_hosts.values()
            if h.compromised_by == 'None' and h.status == 'online'
        )
        r = (healthy / total) * self.HEALTH_BONUS_SCALE

        # Big penalty if any controller is compromised — load-bearing for the grid.
        if any(
            h.is_domain_controller and h.compromised_by != 'None'
            for h in state.all_hosts.values()
        ):
            r += self.CONTROLLER_LOSS_PENALTY

        for attr, ip, v in self._iter_deltas(effect):
            if attr == 'status' and v == 'isolated':
                host = state.all_hosts.get(ip)
                if host and host.compromised_by != 'None':
                    r += 3.0
                else:
                    r -= 1.0
        return r

    def _is_terminal(self, state):
        controllers = [h for h in state.all_hosts.values() if h.is_domain_controller]
        return bool(controllers) and all(
            h.compromised_by != 'None' for h in controllers
        )
