from netforge_rl.diagnostics.base import Diagnostic, DiagnosticResult, flag_host


class DeceptionResistance(Diagnostic):
    """One real foothold and one decoy both raise SIEM alerts. Score Blue on
    isolating the real host without quarantining the trap (deception resistance)."""

    name = 'deception_resistance'
    capability = 'deception'
    scenario_type = 'ransomware'
    max_ticks = 25

    def __init__(self):
        self._real_ip = None
        self._decoy_ip = None

    def setup(self, env):
        live = [
            ip
            for ip, h in env.global_state.all_hosts.items()
            if not ip.startswith('169.254.') and h.status == 'online'
        ]
        ordered = sorted(live)
        self._real_ip = ordered[0]
        self._decoy_ip = ordered[min(1, len(ordered) - 1)]
        if self._decoy_ip == self._real_ip and len(ordered) > 1:
            self._decoy_ip = ordered[1]
        real = env.global_state.all_hosts[self._real_ip]
        real.compromised_by = 'red_operator'
        real.privilege = 'User'
        real.decoy = 'inactive'
        decoy = env.global_state.all_hosts[self._decoy_ip]
        decoy.decoy = 'active'
        decoy.contains_honeytokens = True
        decoy.compromised_by = 'None'
        decoy.privilege = 'None'
        flag_host(env, self._real_ip, severity=8)
        flag_host(env, self._decoy_ip, severity=8)

    def score(self, env, ticks_used):
        real = env.global_state.all_hosts.get(self._real_ip)
        decoy = env.global_state.all_hosts.get(self._decoy_ip)
        isolated_real = real is not None and real.status == 'isolated'
        isolated_decoy = decoy is not None and decoy.status == 'isolated'
        if isolated_real and not isolated_decoy:
            score = 1.0
        elif isolated_real and isolated_decoy:
            score = 0.5
        elif isolated_decoy and not isolated_real:
            score = 0.0
        else:
            score = 0.25
        return DiagnosticResult(
            diagnostic=self.name,
            capability=self.capability,
            policy='',
            score=score,
            details={
                'real_ip': self._real_ip,
                'decoy_ip': self._decoy_ip,
                'isolated_real': isolated_real,
                'isolated_decoy': isolated_decoy,
                'ticks_used': ticks_used,
            },
        )
