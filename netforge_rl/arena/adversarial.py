from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from netforge_rl.arena.evaluate import evaluate_split
from netforge_rl.arena.spec import ARENA_2026
from netforge_rl.baselines.policies import (
    HeuristicRedPolicy,
    KillChainRedPolicy,
    RandomPolicy,
)

DEFAULT_RED_POPULATION: dict[str, Callable] = {
    'random-red': lambda: RandomPolicy(seed=0),
    'heuristic-red': lambda: HeuristicRedPolicy(seed=0),
    'killchain-red': lambda: KillChainRedPolicy(seed=0),
}


def evaluate_population(
    blue_policy_factory: Callable,
    red_population: Mapping[str, Callable] | None = None,
    *,
    scenarios: Sequence[str] = ('ransomware',),
    seeds: Sequence[int] = (0, 1),
    max_ticks: int = 60,
) -> dict:
    """Evaluate one Blue policy against a population of Red opponents."""
    pool = dict(red_population or DEFAULT_RED_POPULATION)
    per_red = {}
    for name, factory in pool.items():
        per_red[name] = evaluate_split(
            factory,
            blue_policy_factory,
            split='dev',
            spec=ARENA_2026,
            scenarios=scenarios,
            seeds=seeds,
            max_ticks=max_ticks,
        )
    sla = [
        v['overall']['mean']['sla_uptime']
        for v in per_red.values()
        if v['overall']['n_episodes']
    ]
    return {
        'n_opponents': len(pool),
        'per_red': per_red,
        'mean_sla_across_red': round(sum(sla) / max(len(sla), 1), 4),
        'worst_sla_across_red': round(min(sla) if sla else 0.0, 4),
    }
