from __future__ import annotations

from collections.abc import Sequence

from netforge_rl.arena.evaluate import run_episode
from netforge_rl.arena.questions.policies import blue, killchain


def run_probe(name: str, *, seeds: Sequence[int], max_ticks: int) -> dict:
    from netforge_rl.diagnostics import run_diagnostic
    from netforge_rl.diagnostics.adaptation import AdaptationShift
    from netforge_rl.diagnostics.deception import DeceptionResistance

    diag = AdaptationShift() if name == 'adaptation' else DeceptionResistance()
    diag.max_ticks = min(int(max_ticks), diag.max_ticks)
    result = run_diagnostic(diag, blue(), seed=int(seeds[0]))
    return {
        'diagnostic': result.diagnostic,
        'capability': result.capability,
        'score': result.score,
        'details': result.details,
    }


def run_constrained(*, seeds: Sequence[int], max_ticks: int) -> dict:
    from netforge_rl.arena.constraints import ConstrainedEnvWrapper

    seed = int(seeds[0])
    unconstrained = run_episode(
        killchain(),
        blue(),
        scenario='ransomware',
        seed=seed,
        max_ticks=max_ticks,
    )
    constrained = run_episode(
        killchain(),
        blue(),
        scenario='ransomware',
        seed=seed,
        max_ticks=max_ticks,
        wrap_env=lambda env: ConstrainedEnvWrapper(env, lambda_cost=1.0),
    )
    return {
        'seed': seed,
        'unconstrained': {
            'sla_uptime': unconstrained.metrics.sla_uptime,
            'false_positives': unconstrained.metrics.false_positives,
            'security': unconstrained.metrics.security,
            'blue_return': round(unconstrained.metrics.blue_return, 4),
        },
        'constrained_lambda_1': {
            'sla_uptime': constrained.metrics.sla_uptime,
            'false_positives': constrained.metrics.false_positives,
            'security': constrained.metrics.security,
            'blue_return': round(constrained.metrics.blue_return, 4),
        },
    }
