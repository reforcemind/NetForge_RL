from netforge_rl.diagnostics.base import Diagnostic, DiagnosticResult, flag_host


class OTKineticResponse(Diagnostic):
    """Plant a foothold on an OT host in the Stuxnet scenario and score Blue on
    isolating it before the PLC suffers kinetic destruction (safety-critical response)."""

    name = 'ot_kinetic_response'
    capability = 'safety'
    scenario_type = 'ot_stuxnet'
    max_ticks = 40

    def __init__(self):
        self._ot_ip = None
        self._isolated = False

    def setup(self, env):
        gs = env.global_state
        ot = [
            ip
            for ip, h in gs.all_hosts.items()
            if gs.get_subnet_name(h.subnet_cidr) == 'OT_Subnet' and h.status == 'online'
        ]
        if not ot:
            ot = [ip for ip in gs.all_hosts if not ip.startswith('169.254.')]
        self._ot_ip = sorted(ot)[0]
        host = gs.all_hosts[self._ot_ip]
        host.compromised_by = 'red_operator'
        host.privilege = 'User'
        flag_host(env, self._ot_ip, severity=10)
        self._isolated = False

    def early_stop(self, env):
        host = env.global_state.all_hosts.get(self._ot_ip)
        if host is not None and host.status == 'isolated':
            self._isolated = True
            return True
        return False

    def score(self, env, ticks_used):
        kinetic = any(
            getattr(h, 'system_integrity', 'clean') == 'kinetic_destruction'
            for h in env.global_state.all_hosts.values()
        )
        score = 1.0 if (self._isolated and not kinetic) else 0.0
        return DiagnosticResult(
            diagnostic=self.name,
            capability=self.capability,
            policy='',
            score=score,
            details={
                'ot_ip': self._ot_ip,
                'isolated': self._isolated,
                'kinetic_destruction': kinetic,
                'ticks_used': ticks_used,
            },
        )
