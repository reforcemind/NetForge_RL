from netforge_rl.diagnostics.base import Diagnostic, DiagnosticResult

PADDING_SUBNET = '169.254.0.0/16'


class FalsePositiveRestraint(Diagnostic):
    """The network is fully clean. Score Blue on restraint — a good defender must not
    quarantine healthy hosts when there is no real threat (precision / false-alarm cost)."""

    name = 'false_positive_restraint'
    capability = 'precision'
    scenario_type = 'ransomware'
    max_ticks = 20

    def setup(self, env):
        for h in env.global_state.all_hosts.values():
            h.compromised_by = 'None'
            if h.privilege in ('User', 'Root'):
                h.privilege = 'None'

    def score(self, env, ticks_used):
        active = [
            h
            for h in env.global_state.all_hosts.values()
            if h.subnet_cidr != PADDING_SUBNET
        ]
        total = max(len(active), 1)
        wrongly_isolated = sum(
            1 for h in active if h.status == 'isolated' and h.compromised_by == 'None'
        )
        score = max(0.0, 1.0 - wrongly_isolated / total)
        return DiagnosticResult(
            diagnostic=self.name,
            capability=self.capability,
            policy='',
            score=score,
            details={
                'wrongly_isolated': wrongly_isolated,
                'active_hosts': total,
                'ticks_used': ticks_used,
            },
        )
