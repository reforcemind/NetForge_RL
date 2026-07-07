import numpy as np


class ObservationMixin:
    """Blue shared-comm channel and per-agent adjacency masking for NetForgeRLEnv."""

    def _build_blue_comm(self) -> np.ndarray:
        """100-dim shared situational awareness from SIEM incidents and isolation state."""
        comm = np.zeros(100, dtype=np.float32)
        ordered = sorted(self.global_state.all_hosts.keys())
        ip_to_idx = {ip: i for i, ip in enumerate(ordered[:100])}

        for entry, _ in self.global_state.siem_log_buffer:
            if not isinstance(entry, str) or '[INCIDENT]' not in entry:
                continue
            for token in entry.split():
                if token.startswith('target='):
                    ip = token[7:]
                    if ip in ip_to_idx:
                        comm[ip_to_idx[ip]] = 1.0
                    break

        recent_subnets = {sub for _, sub in self.global_state.siem_log_buffer[-8:]}
        for i, ip in enumerate(ordered[:100]):
            host = self.global_state.all_hosts[ip]
            if host.privilege in ('User', 'Root') and comm[i] < 0.75:
                comm[i] = max(comm[i], 0.75)
            if host.status == 'isolated':
                comm[i] = max(comm[i], 0.5)
            if host.subnet_cidr in recent_subnets and comm[i] < 0.25:
                comm[i] = 0.25
        return comm

    def _get_adj_matrix_for(self, agent: str) -> np.ndarray:
        """Adjacency matrix masked to the agent's discovered knowledge (fog of war)."""
        full_adj = self.global_state.get_adjacency_matrix()
        if 'blue' in agent.lower():
            return full_adj
        known_ips = self.global_state.agent_knowledge.get(agent, set())
        sorted_ips = sorted(self.global_state.all_hosts.keys())[:100]
        mask = np.zeros(100, dtype=np.float32)
        for i, ip in enumerate(sorted_ips):
            if ip in known_ips:
                mask[i] = 1.0
        return full_adj * mask[None, :] * mask[:, None]
