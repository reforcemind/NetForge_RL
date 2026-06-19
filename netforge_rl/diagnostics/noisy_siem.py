from netforge_rl.diagnostics.base import Diagnostic, DiagnosticResult

class NoisySIEM(Diagnostic):
    """Grade Blue on isolating one compromise amid heavy SIEM noise."""

    name = 'noisy_siem'
    capability = 'attention'
    scenario_type = 'ransomware'
    max_ticks = 25

    def __init__(self):
        self._real_ip = None
        self._isolated_real = False
        self._false_positives = 0
        self._seen_fp = set()

    def setup(self, env):
        non_pad = [
            ip
            for ip in env.global_state.all_hosts
            if not ip.startswith('169.254.')
            and env.global_state.all_hosts[ip].status == 'online'
        ]
        self._real_ip = sorted(non_pad)[len(non_pad) // 2]
        host = env.global_state.all_hosts[self._real_ip]
        host.compromised_by = 'red_operator'
        host.privilege = 'User'
        self._isolated_real = False
        self._false_positives = 0
        self._seen_fp = set()
        if hasattr(env.green_agent, 'noise_rate'):
            env.green_agent.noise_rate = 0.9

    def early_stop(self, env):
        for ip, host in env.global_state.all_hosts.items():
            if (
                host.status == 'isolated'
                and ip != self._real_ip
                and ip not in self._seen_fp
                and not ip.startswith('169.254.')
                and env.global_state.all_hosts[ip].compromised_by == 'None'
            ):
                self._seen_fp.add(ip)
                self._false_positives += 1
        real = env.global_state.all_hosts.get(self._real_ip)
        if real is not None and real.status == 'isolated':
            self._isolated_real = True
            return True
        return False

    def score(self, env, ticks_used):
        base = 1.0 if self._isolated_real else 0.0
        score = max(0.0, base - 0.25 * self._false_positives)
        return DiagnosticResult(
            diagnostic=self.name,
            capability=self.capability,
            policy='',
            score=score,
            details={
                'real_ip': self._real_ip,
                'isolated_real': self._isolated_real,
                'false_positives': self._false_positives,
                'ticks_used': ticks_used,
            },
        )
