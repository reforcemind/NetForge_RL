from dataclasses import dataclass, field

from netforge_rl.environment.parallel_env import NetForgeRLEnv


def flag_host(env, ip: str, severity: int = 8) -> None:
    """Emit an observable SIEM IoC alert implicating ``ip``. Routed through the
    logger so it obeys the scenario's log latency, giving blue policies a telemetry
    signal to detect rather than a ground-truth flag to read."""
    host = env.global_state.all_hosts.get(ip)
    subnet = host.subnet_cidr if host else 'unknown'
    env.siem_logger._push_to_buffer(
        {'signature': 'IOC_ALERT', 'target': ip, 'severity': severity},
        subnet,
        env.global_state,
    )


@dataclass
class DiagnosticResult:
    diagnostic: str
    capability: str
    policy: str
    score: float
    details: dict = field(default_factory=dict)


class Diagnostic:
    name = 'diagnostic'
    capability = 'unspecified'
    controlled_agent = 'blue_dmz'
    scenario_type = 'ransomware'
    max_ticks = 50

    def build_env(self, seed=0):
        env = NetForgeRLEnv(
            {'scenario_type': self.scenario_type, 'max_ticks': self.max_ticks}
        )
        env.reset(seed=seed)
        return env

    def setup(self, env):
        """Post-reset state mutation hook (e.g. plant a fact)."""

    def early_stop(self, env):
        return False

    def score(self, env, ticks_used):
        raise NotImplementedError


def run_diagnostic(diag, policy, *, seed=0):
    env = diag.build_env(seed=seed)
    diag.setup(env)

    ticks = 0
    while env.agents and ticks < diag.max_ticks:
        actions = {a: policy.act(env, a) for a in env.agents}
        env.step(actions)
        ticks += 1
        if diag.early_stop(env):
            break

    return diag.score(env, ticks_used=ticks)
