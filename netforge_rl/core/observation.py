from typing import Any, List

import numpy as np


class BaseObservation:
    """Per-agent partial view of the network (fog-of-war + role-specific channels)."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.visible_hosts = {}
        self.detected_anomalies = []
        self.active_sessions = []
        self.objective_vector = np.zeros(5, dtype=np.float32)
        self.network_telemetry = {}
        self.siem_alerts = []

    def update_from_state(self, global_state: Any, action_effects: List[Any]):
        """Filter ``global_state`` down to what this agent can observe."""
        is_blue = 'blue' in self.agent_id.lower()
        is_commander = 'commander' in self.agent_id.lower()
        is_operator = 'operator' in self.agent_id.lower()

        if global_state:
            known_ips = global_state.agent_knowledge.get(self.agent_id, set())
            for ip in known_ips:
                host = global_state.all_hosts.get(ip)
                if host is None:
                    continue
                if is_blue:
                    # Strict POMDP — Blue can't see physical truth, only SIEM-derived signals.
                    self.visible_hosts[ip] = {
                        'state': 'unknown',
                        'status': host.status,
                        'decoy': host.decoy,
                    }
                else:
                    self.visible_hosts[ip] = {
                        'state': (
                            'compromised' if host.privilege in ('User', 'Root')
                            else 'clean'
                        ),
                        'status': host.status,
                        'decoy': 'unknown',
                    }

        if is_commander or is_blue:
            buffer = getattr(global_state, 'siem_log_buffer', None) or []
            current_tick = getattr(global_state, 'current_tick', 0)
            for log in buffer:
                arrival = log.get('arrival_tick', 0) if isinstance(log, dict) else 0
                if arrival <= current_tick:
                    self.siem_alerts.append(log)

            self.network_telemetry['global_alert_level'] = np.random.uniform(0, 1)
            self.network_telemetry['total_isolated_subnets'] = np.random.randint(0, 5)
            self.network_telemetry['active_alerts'] = len(self.siem_alerts)

        if is_operator:
            self.objective_vector[2] = 1.0

    def to_numpy(self, max_size: int = 256) -> np.ndarray:
        """Pack observation into a fixed-size float32 vector for the policy network."""
        vector = np.zeros(max_size, dtype=np.float32)
        idx = 0

        scalars = []
        if 'global_alert_level' in self.network_telemetry:
            scalars.append(self.network_telemetry['global_alert_level'])
        if 'total_isolated_subnets' in self.network_telemetry:
            scalars.append(self.network_telemetry['total_isolated_subnets'] / 10.0)
        if 'active_alerts' in self.network_telemetry:
            scalars.append(min(self.network_telemetry['active_alerts'] / 20.0, 1.0))

        for val in scalars + list(self.objective_vector):
            if idx >= max_size:
                return vector
            vector[idx] = float(val)
            idx += 1

        for ip in sorted(self.visible_hosts.keys()):
            if idx + 2 >= max_size:
                break
            data = self.visible_hosts[ip]
            state_val = (
                1.0 if data.get('state') == 'compromised'
                else -1.0 if data.get('state') == 'clean'
                else 0.0
            )
            vector[idx] = float(ip.split('.')[-1]) / 255.0
            vector[idx + 1] = state_val
            idx += 2

        return vector
