from netforge_rl.diagnostics.base import Diagnostic, DiagnosticResult
from netforge_rl.environment.parallel_env import NetForgeRLEnv


class DelayedTelemetry(Diagnostic):
    """Plant a Red foothold, delay the SIEM feed, and score Blue on containing it
    despite lagging telemetry (temporal reasoning under partial observability)."""

    name = 'delayed_telemetry'
    capability = 'temporal'
    scenario_type = 'ransomware'
    max_ticks = 30
    log_latency = 6

    def __init__(self):
        self._planted_ip = None
        self._iso_tick = None
        self._tick = 0

    def build_env(self, seed=0):
        env = NetForgeRLEnv(
            {
                'scenario_type': self.scenario_type,
                'max_ticks': self.max_ticks,
                'log_latency': self.log_latency,
            }
        )
        env.reset(seed=seed)
        return env

    def setup(self, env):
        candidates = [
            ip
            for ip, h in env.global_state.all_hosts.items()
            if not ip.startswith('169.254.') and h.status == 'online'
        ]
        self._planted_ip = sorted(candidates)[0]
        host = env.global_state.all_hosts[self._planted_ip]
        host.compromised_by = 'red_operator'
        host.privilege = 'User'
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
                'log_latency': self.log_latency,
                'ticks_used': ticks_used,
            },
        )
