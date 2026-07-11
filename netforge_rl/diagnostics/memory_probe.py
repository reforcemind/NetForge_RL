from netforge_rl.diagnostics.base import Diagnostic, DiagnosticResult, flag_host


class MemoryProbe(Diagnostic):
    """Plant a Red foothold at reset; score Blue on isolating that host fast."""

    name = 'memory_probe'
    capability = 'memory'
    scenario_type = 'ransomware'
    max_ticks = 20

    def __init__(self, planted_subnet_prefix='10.0.0.'):
        self.planted_subnet_prefix = planted_subnet_prefix
        self._planted_ip = None
        self._iso_tick = None
        self._tick = 0

    def setup(self, env):
        candidates = [
            ip
            for ip, h in env.global_state.all_hosts.items()
            if ip.startswith(self.planted_subnet_prefix) and h.status == 'online'
        ]
        if not candidates:
            candidates = [
                ip for ip in env.global_state.all_hosts if not ip.startswith('169.254.')
            ]
        self._planted_ip = sorted(candidates)[0]
        host = env.global_state.all_hosts[self._planted_ip]
        host.compromised_by = 'red_operator'
        host.privilege = 'User'
        flag_host(env, self._planted_ip)
        self._iso_tick = None
        self._tick = 0

    def early_stop(self, env):
        self._tick += 1
        host = env.global_state.all_hosts.get(self._planted_ip)
        if host is not None and host.status == 'isolated' and self._iso_tick is None:
            self._iso_tick = self._tick
            return True
        return False

    def score(self, env, ticks_used):
        if self._iso_tick is None:
            score = 0.0
        else:
            score = max(0.0, 1.0 - (self._iso_tick - 1) / max(self.max_ticks - 1, 1))
        return DiagnosticResult(
            diagnostic=self.name,
            capability=self.capability,
            policy='',
            score=score,
            details={
                'planted_ip': self._planted_ip,
                'isolation_tick': self._iso_tick,
                'ticks_used': ticks_used,
            },
        )
