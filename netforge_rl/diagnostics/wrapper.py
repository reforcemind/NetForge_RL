from typing import Any, List
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

from netforge_rl.core.observation import BaseObservation


class OracleObservation(BaseObservation):
    """Perfect information view of the network state (No fog-of-war)."""

    def update_from_state(self, global_state: Any, _action_effects: List[Any]):
        if global_state:
            # Oracle knows ALL ips
            for ip, host in global_state.all_hosts.items():
                self.visible_hosts[ip] = {
                    'state': 'compromised'
                    if host.privilege in ('User', 'Root')
                    else 'clean',
                    'status': host.status,
                    'decoy': host.decoy,
                }

        buffer = getattr(global_state, 'siem_log_buffer', None) or []
        current_tick = getattr(global_state, 'current_tick', 0)
        for log in buffer:
            arrival = log.get('arrival_tick', 0) if isinstance(log, dict) else 0
            if arrival <= current_tick:
                self.siem_alerts.append(log)
        self.network_telemetry['global_alert_level'] = 1.0  # Perfect clarity
        self.network_telemetry['total_isolated_subnets'] = 0
        self.network_telemetry['active_alerts'] = len(self.siem_alerts)


class DiagnosticsWrapper(BaseParallelWrapper):
    """PettingZoo Wrapper to compute the empirical Information Asymmetry gap."""

    def reset(self, seed=None, options=None):
        obs, infos = super().reset(seed=seed, options=options)
        return self._inject_oracle(obs, infos)

    def step(self, action):
        obs, rewards, terminations, truncations, infos = super().step(action)
        obs, infos = self._inject_oracle(obs, infos)
        return obs, rewards, terminations, truncations, infos

    def _inject_oracle(self, obs, infos):
        global_state = getattr(self.env.unwrapped, 'global_state', None)
        if not global_state:
            return obs, infos

        for agent in self.agents:
            if agent not in obs or agent not in infos:
                continue

            oracle = OracleObservation(agent)
            oracle.update_from_state(global_state, [])
            oracle_vec = oracle.to_numpy(max_size=256)

            agent_obs = obs[agent]['obs']

            distance = float(np.linalg.norm(oracle_vec - agent_obs))

            infos[agent]['oracle_obs'] = oracle_vec
            infos[agent]['information_asymmetry'] = distance

        return obs, infos
