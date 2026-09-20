from __future__ import annotations

from netforge_rl.environment.config import (
    DIFFICULTIES,
    DIFFICULTY_PRESETS,
    EVAL_SEEDS,
    EnvConfig,
)
from netforge_rl.environment.parallel_env import NetForgeRLEnv

__all__ = [
    'DIFFICULTIES',
    'DIFFICULTY_PRESETS',
    'EVAL_SEEDS',
    'make_config',
    'make_env',
]


def make_config(
    difficulty: str = 'medium',
    scenario_type: str = 'ransomware',
    evaluation: bool = False,
    **overrides,
) -> dict:
    """Env config for a difficulty tier; overrides win over the preset."""
    if difficulty not in DIFFICULTIES:
        raise KeyError(
            f'Unknown difficulty {difficulty!r}. Choose from {sorted(DIFFICULTIES)}.'
        )
    cfg = DIFFICULTIES[difficulty].apply(
        EnvConfig(scenario_type=scenario_type, evaluation_mode=evaluation)
    )
    if overrides:
        cfg = EnvConfig.from_mapping({**cfg.to_dict(), **overrides})
    return cfg.to_dict()


def make_env(
    difficulty: str = 'medium',
    scenario_type: str = 'ransomware',
    evaluation: bool = False,
    seed: int | None = None,
    **overrides,
) -> NetForgeRLEnv:
    """Build (and reset, if seed is given) an env at a difficulty tier."""
    env = NetForgeRLEnv(make_config(difficulty, scenario_type, evaluation, **overrides))
    if seed is not None:
        env.reset(seed=seed)
    return env
