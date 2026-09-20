from __future__ import annotations

from collections.abc import Sequence

from netforge_rl.arena.evaluate import run_episode
from netforge_rl.arena.questions.policies import blue, killchain


def run_reward_design(*, seeds: Sequence[int], max_ticks: int) -> dict:
    from netforge_rl.rewards.variants import RewardDesignWrapper

    variants = ('default', 'sla_only', 'fp_penalized')
    out = {}
    seed = int(seeds[0])
    for variant in variants:
        rec = run_episode(
            killchain(),
            blue(),
            scenario='ransomware',
            seed=seed,
            max_ticks=max_ticks,
            wrap_env=lambda env, v=variant: RewardDesignWrapper(env, variant=v),
        )
        m = rec.metrics
        out[variant] = {
            'isolated_hosts': m.isolated_hosts,
            'false_positives': m.false_positives,
            'sla_uptime': m.sla_uptime,
            'mission_success': m.mission_success,
            'blue_return': round(m.blue_return, 4),
            'raw_vs_shaped': 'see info[reward_variant] / info[raw_reward]',
        }
    return {'seed': seed, 'variants': out}
