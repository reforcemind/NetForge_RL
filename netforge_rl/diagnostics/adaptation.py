from netforge_rl.baselines.policies import KillChainRedPolicy, RandomPolicy
from netforge_rl.diagnostics.base import Diagnostic, DiagnosticResult, flag_host
from netforge_rl.environment.constants import PADDING_SUBNET


class AdaptationShift(Diagnostic):
    """Kill-chain Red for the first half of the episode, then a random spreader.
    Score Blue on SLA and containment *after* the attacker strategy shifts."""

    name = 'adaptation_shift'
    capability = 'adaptation'
    scenario_type = 'ransomware'
    max_ticks = 40
    switch_tick = 20

    def __init__(self, seed=0):
        self._red_a = KillChainRedPolicy(seed=seed)
        self._red_b = RandomPolicy(seed=seed + 1)
        self._switch_compromised = 0
        self._final_sla = 1.0
        self._contained_after = 0
        self._new_infections = 0

    def setup(self, env):
        live = [
            ip
            for ip, h in env.global_state.all_hosts.items()
            if not ip.startswith('169.254.') and h.status == 'online'
        ]
        planted = sorted(live)[0]
        host = env.global_state.all_hosts[planted]
        host.compromised_by = 'red_operator'
        host.privilege = 'User'
        flag_host(env, planted)
        self._switch_compromised = 0
        self._final_sla = 1.0
        self._contained_after = 0
        self._new_infections = 0

    def act_for(self, env, agent_id, blue_policy):
        if 'red' in agent_id.lower():
            if env.current_tick < self.switch_tick:
                return self._red_a.act(env, agent_id)
            return self._red_b.act(env, agent_id)
        return blue_policy.act(env, agent_id)

    def early_stop(self, env):
        if env.current_tick == self.switch_tick:
            self._switch_compromised = self._count_compromised(env)
        return False

    def score(self, env, ticks_used):
        active = [
            h
            for h in env.global_state.all_hosts.values()
            if h.subnet_cidr != PADDING_SUBNET
        ]
        total = max(len(active), 1)
        healthy = sum(
            1 for h in active if h.compromised_by == 'None' and h.status == 'online'
        )
        self._final_sla = healthy / total
        final_comp = self._count_compromised(env)
        self._new_infections = max(0, final_comp - self._switch_compromised)
        isolated_comp = sum(
            1 for h in active if h.compromised_by != 'None' and h.status == 'isolated'
        )
        self._contained_after = isolated_comp
        growth_penalty = min(1.0, self._new_infections / max(total, 1))
        contain = isolated_comp / max(final_comp, 1)
        score = max(0.0, 0.5 * self._final_sla + 0.5 * contain - 0.25 * growth_penalty)
        score = min(1.0, score)
        return DiagnosticResult(
            diagnostic=self.name,
            capability=self.capability,
            policy='',
            score=score,
            details={
                'switch_tick': self.switch_tick,
                'switch_compromised': self._switch_compromised,
                'final_compromised': final_comp,
                'new_infections': self._new_infections,
                'final_sla': self._final_sla,
                'ticks_used': ticks_used,
            },
        )

    @staticmethod
    def _count_compromised(env) -> int:
        return sum(
            1
            for h in env.global_state.all_hosts.values()
            if h.subnet_cidr != PADDING_SUBNET and h.compromised_by != 'None'
        )
