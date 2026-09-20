from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from netforge_rl.arena.metrics import ArenaMetrics, aggregate, metrics_from_env
from netforge_rl.arena.spec import ARENA_2026, ArenaSpec, SplitName
from netforge_rl.environment.graph_wrapper import GraphObservationWrapper
from netforge_rl.scenarios.yaml_dsl import make_env_from_spec, resolve_scenario


@dataclass
class EpisodeRecord:
    scenario: str
    seed: int
    split: str
    metrics: ArenaMetrics
    extras: dict = field(default_factory=dict)


def _act(policy, env, agent_id, obs) -> np.ndarray:
    fn = getattr(policy, 'act', None)
    if fn is None:
        return np.zeros(len(env.action_space(agent_id).nvec), dtype=np.int64)
    mode = getattr(policy, '_nf_act', None)
    if mode == 'obs':
        return np.asarray(fn(obs, agent_id), dtype=np.int64)
    if mode != 'env':
        try:
            out = fn(env, agent_id)
            policy._nf_act = 'env'
            return np.asarray(out, dtype=np.int64)
        except TypeError:
            policy._nf_act = 'obs'
            return np.asarray(fn(obs, agent_id), dtype=np.int64)
    return np.asarray(fn(env, agent_id), dtype=np.int64)


def run_episode(
    red_policy,
    blue_policy,
    *,
    scenario: str = 'ransomware',
    seed: int = 0,
    max_ticks: int = 200,
    evaluation_mode: bool = False,
    difficulty: str = 'medium',
    graph_obs: bool = True,
    env_overrides: dict | None = None,
    wrap_env: Callable | None = None,
) -> EpisodeRecord:
    spec = resolve_scenario(scenario)
    overrides = dict(env_overrides or {})
    overrides.setdefault('max_ticks', max_ticks)
    overrides.setdefault('evaluation_mode', evaluation_mode)
    env = make_env_from_spec(spec, difficulty=difficulty, seed=None, **overrides)
    if graph_obs:
        env = GraphObservationWrapper(env, oracle=False)
    if wrap_env is not None:
        env = wrap_env(env)
    obs, _ = env.reset(seed=seed)
    for pol in (red_policy, blue_policy):
        if hasattr(pol, 'reset'):
            pol.reset()
        if hasattr(pol, 'bind_env'):
            pol.bind_env(env)

    blue_return = red_return = 0.0
    last_info: dict = {}
    while env.agents:
        actions = {}
        for agent_id in env.agents:
            policy = red_policy if 'red' in agent_id else blue_policy
            actions[agent_id] = _act(policy, env, agent_id, obs.get(agent_id, {}))
        obs, rewards, term, trunc, last_info = env.step(actions)
        for agent_id, reward in rewards.items():
            if 'red' in agent_id:
                red_return += float(reward)
            else:
                blue_return += float(reward)
        if all(term.values()) or all(trunc.values()):
            break

    inner = env.unwrapped if hasattr(env, 'unwrapped') else env
    metrics = metrics_from_env(inner, last_info, blue_return, red_return)
    return EpisodeRecord(
        scenario=spec.name,
        seed=seed,
        split='custom',
        metrics=metrics,
    )


def evaluate_split(
    red_policy_factory: Callable,
    blue_policy_factory: Callable,
    *,
    split: SplitName = 'dev',
    spec: ArenaSpec | None = None,
    scenarios: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    max_ticks: int | None = None,
    difficulty: str | None = None,
    graph_obs: bool = True,
    env_overrides: dict | None = None,
    wrap_env: Callable | None = None,
) -> dict:
    arena = spec or ARENA_2026
    split_info = arena.split(split)
    scenarios = tuple(scenarios or arena.scenarios)
    seeds = tuple(seeds or split_info.seeds)
    max_ticks = max_ticks if max_ticks is not None else arena.max_ticks
    difficulty = difficulty or arena.difficulty

    records: list[EpisodeRecord] = []
    per_scenario: dict[str, list[ArenaMetrics]] = {s: [] for s in scenarios}
    for scenario in scenarios:
        for seed in seeds:
            rec = run_episode(
                red_policy_factory(),
                blue_policy_factory(),
                scenario=scenario,
                seed=int(seed),
                max_ticks=max_ticks,
                evaluation_mode=split_info.evaluation_mode,
                difficulty=difficulty,
                graph_obs=graph_obs,
                env_overrides=env_overrides,
                wrap_env=wrap_env,
            )
            rec.split = split
            records.append(rec)
            per_scenario[scenario].append(rec.metrics)

    overall = aggregate([r.metrics for r in records])
    return {
        'arena': arena.version,
        'split': split,
        'public': split_info.public,
        'n_episodes': len(records),
        'scenarios': list(scenarios),
        'overall': overall.as_dict(),
        'per_scenario': {
            name: aggregate(vals).as_dict() for name, vals in per_scenario.items()
        },
    }
