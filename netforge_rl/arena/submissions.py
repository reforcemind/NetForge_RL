"""CAGE-style ``make_blue`` eval. Rank is a card, not mean reward."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from netforge_rl.arena.evaluate import evaluate_split
from netforge_rl.arena.spec import ARENA_2026, SplitName
from netforge_rl.baselines.policies import KillChainRedPolicy

# Local smoke. --official uses the Arena freeze.
LOCAL_EVAL = {
    'split': 'dev',
    'scenarios': ('ransomware',),
    'seeds': (9001, 9002),
    'max_ticks': 80,
}

OFFICIAL_EVAL = {
    'split': 'dev',
    'scenarios': ARENA_2026.scenarios,
    'seeds': ARENA_2026.dev.seeds,
    'max_ticks': ARENA_2026.max_ticks,
}


@dataclass
class Submission:
    name: str
    team: str
    technique: str
    path: Path
    make_blue: Callable
    make_red: Callable | None = None
    wrap: Callable | None = None
    paper: str = ''


def load_submission(path: str | Path) -> Submission:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f'Submission not found: {path}')
    spec = importlib.util.spec_from_file_location('netforge_submission', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import submission {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    make_blue = getattr(mod, 'make_blue', None)
    if make_blue is None:
        raise AttributeError(
            f'{path} must define make_blue(seed=0) -> policy with act(env, agent_id)'
        )
    return Submission(
        name=str(getattr(mod, 'NAME', path.stem)),
        team=str(getattr(mod, 'TEAM', '')),
        technique=str(getattr(mod, 'TECHNIQUE', '')),
        paper=str(getattr(mod, 'PAPER', '')),
        path=path,
        make_blue=make_blue,
        make_red=getattr(mod, 'make_red', None),
        wrap=getattr(mod, 'wrap', None),
    )


def evaluate_submission(
    submission: Submission,
    *,
    split: SplitName = 'dev',
    scenarios: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    max_ticks: int | None = None,
    official: bool = False,
    agent_card: bool = False,
) -> dict:
    proto = dict(OFFICIAL_EVAL if official else LOCAL_EVAL)
    if official:
        split = proto['split']  # type: ignore[assignment]
    scenarios = tuple(scenarios or proto['scenarios'])
    seeds = tuple(seeds or proto['seeds'])
    max_ticks = int(max_ticks if max_ticks is not None else proto['max_ticks'])

    red_factory = submission.make_red or (lambda: KillChainRedPolicy(seed=0))
    wrap_env = submission.wrap
    result = evaluate_split(
        red_factory,
        lambda: submission.make_blue(seed=0),
        split=split,
        scenarios=scenarios,
        seeds=seeds,
        max_ticks=max_ticks,
        wrap_env=wrap_env,
    )
    mean = result['overall'].get('mean', {})
    warnings = ['Rank: mission/SLA/FPs/security/CVaR — not mean reward.']
    if wrap_env is not None:
        warnings.append('wrap() runs after belief graphs; do not attach oracle_graph.')
    payload = {
        'name': submission.name,
        'team': submission.team,
        'technique': submission.technique,
        'paper': submission.paper,
        'path': str(submission.path),
        'arena': result['arena'],
        'split': result['split'],
        'public': result['public'],
        'official': official,
        'n_episodes': result['n_episodes'],
        'scenarios': result['scenarios'],
        'overall': result['overall'],
        'per_scenario': result['per_scenario'],
        'headline': {
            'mission_success': mean.get('mission_success'),
            'sla_uptime': mean.get('sla_uptime'),
            'security': mean.get('security'),
            'false_positives': mean.get('false_positives'),
            'catastrophic_failure': mean.get('catastrophic_failure'),
            'cvar05_blue_return': result['overall']
            .get('cvar05', {})
            .get('blue_return'),
            'mean_blue_return': mean.get('blue_return'),
        },
        'warnings': warnings,
    }
    if agent_card:
        from netforge_rl.arena.agent_card import build_agent_card

        payload['agent_card'] = build_agent_card(
            lambda: submission.make_blue(seed=0),
            name=submission.name,
            seeds=tuple(seeds)[:2],
            run_arena=False,
        )
    return payload
