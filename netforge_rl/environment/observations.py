import numpy as np

from netforge_rl.environment.constants import BLUE_COMM_DIM, N_HOST_SLOTS, SIEM_ENCODE_N


class ObservationMixin:
    """Blue shared-comm channel and per-agent adjacency masking for NetForgeRLEnv."""

    def _build_blue_comm(self) -> np.ndarray:
        comm = np.zeros(BLUE_COMM_DIM, dtype=np.float32)
        ordered = sorted(self.global_state.all_hosts.keys())
        ip_to_idx = {ip: i for i, ip in enumerate(ordered[:N_HOST_SLOTS])}

        for entry, _ in self.global_state.siem_log_buffer:
            if not isinstance(entry, str) or '[INCIDENT]' not in entry:
                continue
            for token in entry.split():
                if token.startswith('target='):
                    ip = token[7:]
                    if ip in ip_to_idx:
                        comm[ip_to_idx[ip]] = 1.0
                    break

        recent_subnets = {
            sub for _, sub in self.global_state.siem_log_buffer[-SIEM_ENCODE_N:]
        }
        for i, ip in enumerate(ordered[:N_HOST_SLOTS]):
            host = self.global_state.all_hosts[ip]
            if host.status == 'isolated':
                comm[i] = max(comm[i], 0.5)
            if host.subnet_cidr in recent_subnets and comm[i] < 0.25:
                comm[i] = 0.25
        return comm

    def _get_adj_matrix_for(self, agent: str) -> np.ndarray:
        """Adjacency masked to discovered hosts."""
        full_adj = self.global_state.get_adjacency_matrix()
        if 'blue' in agent.lower():
            return full_adj
        known_ips = self.global_state.agent_knowledge.get(agent, set())
        sorted_ips = sorted(self.global_state.all_hosts.keys())[:N_HOST_SLOTS]
        mask = np.zeros(N_HOST_SLOTS, dtype=np.float32)
        for i, ip in enumerate(sorted_ips):
            if ip in known_ips:
                mask[i] = 1.0
        return full_adj * mask[None, :] * mask[:, None]
