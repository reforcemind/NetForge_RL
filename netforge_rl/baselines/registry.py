from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from netforge_rl.arena.spec import ARENA_2026
from netforge_rl.baselines.policies import (
    HeuristicBluePolicy,
    HeuristicRedPolicy,
    KillChainRedPolicy,
    RandomPolicy,
)

# Runnable here. MAPPO / QMIX / GNN stay out of tree; protocol names still list them.
RUNNABLE: dict[str, Callable] = {
    'random': lambda seed=0: RandomPolicy(seed=seed),
    'heuristic-blue': lambda seed=0: HeuristicBluePolicy(seed=seed),
    'heuristic-red': lambda seed=0: HeuristicRedPolicy(seed=seed),
    'killchain-red': lambda seed=0: KillChainRedPolicy(seed=seed),
}

LEARNED_STUBS = {
    'ippo': 'netforge_rl.baselines.jax_ppo (extra: jax). Train then pass an .npz.',
    'mappo': 'RLlib via netforge_rl.bridges.rllib_bridge / benchmarks/rllib_rmappo.py',
    'rmappo': 'benchmarks/rllib_rmappo.py (LSTM MAPPO). Keep the algorithm outside.',
    'gnn-mappo': 'Train on info["graph"] belief graphs. Not bundled; evaluate here.',
    'qmix': 'Use a QMIX implementation (e.g. JaxMARL / RLlib) against this gym.',
    'llm': 'netforge_rl.semantic runner. Prompt-only / SFT / RL-LLM compared in Arena.',
}


def make_policy(name: str, seed: int = 0):
    key = name.lower()
    if key in RUNNABLE:
        return RUNNABLE[key](seed=seed)
    path = Path(name)
    if path.suffix == '.npz':
        from netforge_rl.baselines.jax_ppo import load_params

        return load_params(str(path))
    if key in LEARNED_STUBS:
        raise FileNotFoundError(
            f'{name} is a reference baseline in the Arena protocol, not a bundled '
            f'checkpoint. {LEARNED_STUBS[key]}'
        )
    if path.exists():
        raise FileNotFoundError(
            f'No built-in loader for {path.suffix} checkpoints. Wrap your policy '
            f'in a BasePolicy with act(env, agent_id) and pass --policy heuristic-blue '
            f'or a Python factory. See docs/baselines.md.'
        )
    raise KeyError(
        f'Unknown policy {name!r}. Runnable: {sorted(RUNNABLE)}. '
        f'Protocol names: {list(ARENA_2026.baselines)}'
    )
