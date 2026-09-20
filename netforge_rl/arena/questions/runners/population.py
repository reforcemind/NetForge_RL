from __future__ import annotations

from collections.abc import Sequence

from netforge_rl.arena.questions.policies import blue, killchain
from netforge_rl.baselines.policies import HeuristicRedPolicy, RandomPolicy


def run_red_population(*, seeds: Sequence[int], max_ticks: int) -> dict:
    from netforge_rl.arena.adversarial import evaluate_population

    return evaluate_population(
        blue,
        {
            'random-red': lambda: RandomPolicy(seed=0),
            'heuristic-red': lambda: HeuristicRedPolicy(seed=0),
            'killchain-red': killchain,
        },
        scenarios=('ransomware',),
        seeds=tuple(seeds)[:2],
        max_ticks=max_ticks,
    )
