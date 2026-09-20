from __future__ import annotations

from collections.abc import Sequence

from netforge_rl.arena.evaluate import run_episode
from netforge_rl.arena.questions.policies import blue, killchain


def run_time_mode(*, seeds: Sequence[int], max_ticks: int) -> dict:
    seed = int(seeds[0])
    out = {}
    for mode in ('event', 'fixed'):
        rec = run_episode(
            killchain(),
            blue(),
            scenario='ransomware',
            seed=seed,
            max_ticks=max_ticks,
            env_overrides={'time_mode': mode},
        )
        out[mode] = {
            'steps': rec.metrics.steps,
            'sla_uptime': rec.metrics.sla_uptime,
            'blue_return': round(rec.metrics.blue_return, 4),
            'security': rec.metrics.security,
        }
    return {'seed': seed, 'modes': out}


def run_ood_topology(*, seeds: Sequence[int], max_ticks: int) -> dict:
    seed = int(seeds[0])
    out = {}
    for name, overrides in (
        ('medium_default', None),
        ('larger_hosts', {'max_active_hosts': 40}),
    ):
        rec = run_episode(
            killchain(),
            blue(),
            scenario='ransomware',
            seed=seed,
            max_ticks=max_ticks,
            env_overrides=overrides,
        )
        out[name] = {
            'sla_uptime': rec.metrics.sla_uptime,
            'security': rec.metrics.security,
            'compromised_hosts': rec.metrics.compromised_hosts,
            'isolated_hosts': rec.metrics.isolated_hosts,
        }
    return {'seed': seed, 'topologies': out}
