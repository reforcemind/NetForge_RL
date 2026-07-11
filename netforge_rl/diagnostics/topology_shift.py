from netforge_rl.diagnostics.base import Diagnostic, DiagnosticResult, flag_host
from netforge_rl.environment.parallel_env import NetForgeRLEnv


class TopologyShift(Diagnostic):
    """Plant a foothold while the surrounding network churns (hosts arrive and leave),
    and score Blue on isolating the planted host despite the shifting topology
    (generalization to non-stationary networks)."""

    name = 'topology_shift'
    capability = 'generalization'
    scenario_type = 'ransomware'
    max_ticks = 30

    def __init__(self):
        self._planted_ip = None
        self._iso_tick = None
        self._tick = 0

    def build_env(self, seed=0):
        env = NetForgeRLEnv(
            {
                'scenario_type': self.scenario_type,
                'max_ticks': self.max_ticks,
                'topology_churn_rate': 0.05,
                'topology_arrival_rate': 0.02,
                'topology_migration_rate': 0.0,
                'dhcp_interval': 0,
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
