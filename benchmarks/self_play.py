from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from benchmarks.competition_eval import PolicyAgent, RandomAgent, run_episode
from netforge_rl.baselines.policies import (
    HeuristicBluePolicy,
    HeuristicRedPolicy,
    KillChainRedPolicy,
)

RESULTS_DIR = Path(__file__).parent / 'results'

# Population of opponent policies. Each entry builds a fresh competition Agent.
RED_POOL: dict[str, Callable] = {
    'random-red': lambda: RandomAgent(seed=0),
    'heuristic-red': lambda: PolicyAgent(HeuristicRedPolicy(seed=0)),
    'killchain-red': lambda: PolicyAgent(KillChainRedPolicy(seed=0)),
}
BLUE_POOL: dict[str, Callable] = {
    'random-blue': lambda: RandomAgent(seed=0),
    'heuristic-blue': lambda: PolicyAgent(HeuristicBluePolicy(seed=0)),
}


def _expected(r_a: float, r_b: float) -> float:
    """Standard logistic Elo expectation that A scores against B."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def run_match(
    red_factory: Callable,
    blue_factory: Callable,
    scenarios: list[str],
    seeds: list[int],
    max_ticks: int,
) -> float:
    """Play a red/blue match; return Blue's score share (mean SLA uptime) in [0, 1].
    SLA is the shared axis: S_blue = mean SLA, S_red = 1 - S_blue (fractional Elo)."""
    slas = []
    for scenario in scenarios:
        for seed in seeds:
            ep = run_episode(
                red_factory(),
                blue_factory(),
                scenario=scenario,
                seed=seed,
                max_ticks=max_ticks,
            )
            slas.append(ep.sla_uptime)
    return sum(slas) / max(len(slas), 1)


def population_tournament(
    red_pool: dict[str, Callable] = RED_POOL,
    blue_pool: dict[str, Callable] = BLUE_POOL,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    max_ticks: int = 150,
    k: float = 32.0,
    base_rating: float = 1000.0,
) -> dict:
    """Round-robin every red against every blue; rate all policies on one shared Elo
    ladder (red and blue compete over the same SLA quantity)."""
    scenarios = scenarios or ['ransomware', 'apt_espionage', 'cloud_hybrid']
    seeds = seeds or list(range(5))
    ratings = {name: base_rating for name in {**red_pool, **blue_pool}}
    matches = []

    for red_name, red_factory in red_pool.items():
        for blue_name, blue_factory in blue_pool.items():
            s_blue = run_match(red_factory, blue_factory, scenarios, seeds, max_ticks)
            s_red = 1.0 - s_blue
            exp_blue = _expected(ratings[blue_name], ratings[red_name])
            ratings[blue_name] += k * (s_blue - exp_blue)
            ratings[red_name] += k * (s_red - (1.0 - exp_blue))
            matches.append(
                {
                    'red': red_name,
                    'blue': blue_name,
                    'blue_sla': round(s_blue, 4),
                    'red_score': round(s_red, 4),
                }
            )
            print(
                f'{red_name:>14}  vs  {blue_name:<14}  '
                f'blue_sla={s_blue:.3f}  red={s_red:.3f}'
            )

    ladder = sorted(
        ({'policy': n, 'rating': round(r, 1)} for n, r in ratings.items()),
        key=lambda e: e['rating'],
        reverse=True,
    )
    result = {
        'scenarios': scenarios,
        'n_seeds': len(seeds),
        'max_ticks': max_ticks,
        'ladder': ladder,
        'matches': matches,
    }
    print('\n=== Elo ladder ===')
    for rank, e in enumerate(ladder, 1):
        print(f'{rank:>2}. {e["policy"]:<16} {e["rating"]:>7.1f}')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--max-ticks', type=int, default=150)
    p.add_argument('--scenarios', nargs='+', default=None)
    args = p.parse_args()

    res = population_tournament(
        scenarios=args.scenarios,
        seeds=list(range(args.seeds)),
        max_ticks=args.max_ticks,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'self_play_elo.json').write_text(json.dumps(res, indent=2))
    print(f'\nSaved: {RESULTS_DIR / "self_play_elo.json"}')
