from dataclasses import dataclass, field

from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.diagnostics.delayed_telemetry import DelayedTelemetry
from netforge_rl.diagnostics.false_positive import FalsePositiveRestraint
from netforge_rl.diagnostics.memory_probe import MemoryProbe
from netforge_rl.diagnostics.noisy_siem import NoisySIEM
from netforge_rl.diagnostics.ot_kinetic import OTKineticResponse
from netforge_rl.diagnostics.topology_shift import TopologyShift


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


def all_diagnostics():
    return [
        MemoryProbe(),
        NoisySIEM(),
        DelayedTelemetry(),
        FalsePositiveRestraint(),
        OTKineticResponse(),
        TopologyShift(),
    ]
