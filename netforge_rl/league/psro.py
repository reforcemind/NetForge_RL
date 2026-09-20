from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from netforge_rl.arena.evaluate import run_episode


def payoff_matrix(
    red_population: Mapping[str, Callable],
    blue_population: Mapping[str, Callable],
    *,
    scenarios: Sequence[str] = ('ransomware',),
    seeds: Sequence[int] = (0,),
    max_ticks: int = 40,
    metric: str = 'sla_uptime',
) -> dict:
    """Empirical Red vs Blue payoff for PSRO-style evaluation (not Elo-only)."""
    red_names = list(red_population)
    blue_names = list(blue_population)
    matrix = [[0.0 for _ in blue_names] for _ in red_names]
    for i, red_name in enumerate(red_names):
        for j, blue_name in enumerate(blue_names):
            values = []
            for scenario in scenarios:
                for seed in seeds:
                    rec = run_episode(
                        red_population[red_name](),
                        blue_population[blue_name](),
                        scenario=scenario,
                        seed=int(seed),
                        max_ticks=max_ticks,
                    )
                    values.append(float(getattr(rec.metrics, metric)))
            matrix[i][j] = round(sum(values) / max(len(values), 1), 4)
    return {
        'metric': metric,
        'red': red_names,
        'blue': blue_names,
        'payoff_blue': matrix,
        'note': 'Entry [r][b] is Blue metric vs Red. Use as a PSRO empirical game.',
    }
