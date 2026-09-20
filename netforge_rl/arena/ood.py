from __future__ import annotations

from collections.abc import Callable, Sequence

from netforge_rl.arena.evaluate import evaluate_split
from netforge_rl.arena.spec import ARENA_2026

OOD_TASKS: tuple[dict, ...] = (
    {
        'name': 'unseen_topology_size',
        'env_overrides': {'max_active_hosts': 40},
        'scenarios': ('ransomware',),
        'description': 'Train on medium, evaluate on a larger live host count.',
    },
    {
        'name': 'unseen_attacker',
        'red': 'killchain-red',
        'scenarios': ('apt_espionage',),
        'description': 'Kill-chain Red if the policy was trained vs heuristic.',
    },
    {
        'name': 'unseen_telemetry',
        'env_overrides': {'log_latency': 8},
        'scenarios': ('ransomware',),
        'description': 'Delayed SIEM relative to the training latency.',
    },
    {
        'name': 'unseen_scenario',
        'scenarios': ('ot_stuxnet',),
        'description': 'Held-out scenario family (kinetic OT).',
    },
)


def evaluate_ood(
    red_policy_factory: Callable,
    blue_policy_factory: Callable,
    *,
    tasks: Sequence[dict] = OOD_TASKS,
    seeds: Sequence[int] = (0, 1),
    max_ticks: int = 60,
    difficulty: str = 'medium',
) -> dict:
    """Train-distribution vs unseen size / attacker / telemetry / scenario."""
    results = {}
    for task in tasks:
        red_factory = red_policy_factory
        if task.get('red') == 'killchain-red':
            from netforge_rl.baselines.policies import KillChainRedPolicy

            def red_factory():
                return KillChainRedPolicy(seed=0)

        results[task['name']] = evaluate_split(
            red_factory,
            blue_policy_factory,
            split='dev',
            spec=ARENA_2026,
            scenarios=task.get('scenarios'),
            seeds=seeds,
            max_ticks=max_ticks,
            difficulty=difficulty,
            env_overrides=task.get('env_overrides'),
        )
        results[task['name']]['description'] = task['description']
    return {'tasks': results}
