from .base_env import BaseNetForgeRLEnv
from .config import DIFFICULTIES, DIFFICULTY_PRESETS, EVAL_SEEDS, EnvConfig
from .parallel_env import NetForgeRLEnv
from .presets import make_config, make_env

__all__ = [
    'DIFFICULTIES',
    'DIFFICULTY_PRESETS',
    'EVAL_SEEDS',
    'BaseNetForgeRLEnv',
    'EnvConfig',
    'NetForgeRLEnv',
    'NetForgeSingleAgentEnv',
    'make_config',
    'make_env',
    'register_envs',
]


def __getattr__(name):
    if name == 'NetForgeSingleAgentEnv':
        from .gym_env import NetForgeSingleAgentEnv

        return NetForgeSingleAgentEnv
    if name == 'register_envs':
        from .registry import register_envs

        return register_envs
    raise AttributeError(name)
