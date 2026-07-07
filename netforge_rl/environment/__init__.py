from .base_env import BaseNetForgeRLEnv
from .parallel_env import NetForgeRLEnv
from .presets import (
    DIFFICULTY_PRESETS,
    EVAL_SEEDS,
    make_config,
    make_env,
)

__all__ = [
    'BaseNetForgeRLEnv',
    'NetForgeRLEnv',
    'DIFFICULTY_PRESETS',
    'EVAL_SEEDS',
    'make_config',
    'make_env',
    'NetForgeSingleAgentEnv',
]


def __getattr__(name):
    if name == 'NetForgeSingleAgentEnv':
        from .gym_env import NetForgeSingleAgentEnv

        return NetForgeSingleAgentEnv
    raise AttributeError(name)
