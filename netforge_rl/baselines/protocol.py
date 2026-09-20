from __future__ import annotations

from collections.abc import Callable, Sequence

from netforge_rl.arena.evaluate import evaluate_split
from netforge_rl.arena.spec import ARENA_2026, ArenaSpec
from netforge_rl.baselines.registry import RUNNABLE, make_policy


def run_baseline_table(
    names: Sequence[str] | None = None,
    *,
    spec: ArenaSpec | None = None,
    split: str = 'dev',
    scenarios: Sequence[str] = ('ransomware',),
    seeds: Sequence[int] = (0, 1),
    max_ticks: int = 40,
    red_name: str = 'killchain-red',
) -> dict:
    """Every algorithm under the same seeds, scenarios, budget, and metrics."""
    arena = spec or ARENA_2026
    names = list(names or sorted(RUNNABLE))
    rows = []
    for name in names:
        if name not in RUNNABLE and not name.endswith(('.npz', '.pt', '.json')):
            rows.append(
                {
                    'algorithm': name,
                    'status': 'not_bundled',
                    'note': 'Train separately; evaluate with this protocol.',
                }
            )
            continue
        blue_factory: Callable
        red_factory: Callable
        if 'red' in name:

            def red_factory(n=name):
                return make_policy(n)

            def blue_factory():
                return make_policy('heuristic-blue')

            team = 'red'
        else:

            def red_factory():
                return make_policy(red_name)

            def blue_factory(n=name):
                return make_policy(n)

            team = 'blue'
        result = evaluate_split(
            red_factory,
            blue_factory,
            split=split,  # type: ignore[arg-type]
            spec=arena,
            scenarios=scenarios,
            seeds=seeds,
            max_ticks=max_ticks,
        )
        mean = result['overall']['mean']
        rows.append(
            {
                'algorithm': name,
                'team': team,
                'status': 'ok',
                'mission_success': mean.get('mission_success'),
                'sla_uptime': mean.get('sla_uptime'),
                'security': mean.get('security'),
                'false_positives': mean.get('false_positives'),
                'catastrophic_failure': mean.get('catastrophic_failure'),
                'cvar05_blue_return': result['overall']['cvar05'].get('blue_return'),
                'n_episodes': result['n_episodes'],
            }
        )
    return {
        'arena': arena.version,
        'split': split,
        'seeds': list(seeds),
        'scenarios': list(scenarios),
        'max_ticks': max_ticks,
        'compute_budget_env_steps': arena.compute_budget_env_steps,
        'rows': rows,
    }


def format_baseline_table(table: dict) -> str:
    header = (
        f'{"algorithm":<18}{"mission":>10}{"SLA":>8}{"security":>10}'
        f'{"FP":>8}{"CVaR":>10}'
    )
    lines = [header]
    for row in table['rows']:
        if row.get('status') != 'ok':
            lines.append(f'{row["algorithm"]:<18}  ({row.get("status")})')
            continue
        lines.append(
            f'{row["algorithm"]:<18}'
            f'{row["mission_success"]:10.2f}'
            f'{row["sla_uptime"]:8.2f}'
            f'{row["security"]:10.2f}'
            f'{row["false_positives"]:8.2f}'
            f'{row["cvar05_blue_return"]:10.2f}'
        )
    return '\n'.join(lines)
