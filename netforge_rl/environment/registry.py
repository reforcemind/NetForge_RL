"""Gymnasium and PettingZoo registration — ``gym.make`` / ``pettingzoo.make``."""

from __future__ import annotations

GYM_BLUE_ID = 'NetForge/Blue-v4'
GYM_IDS = (GYM_BLUE_ID, 'NetForge-v4')
PZ_RANSOMWARE_ID = 'netforge/ransomware-v4'
PZ_ARENA_ID = 'netforge/arena-v4'
PZ_IDS = (PZ_RANSOMWARE_ID, PZ_ARENA_ID)


def make_parallel_env(**kwargs):
    from netforge_rl.environment.parallel_env import NetForgeRLEnv

    cfg = {'scenario_type': 'ransomware', 'max_ticks': 200, **kwargs}
    cfg.pop('max_cycles', None)
    return NetForgeRLEnv(cfg)


def make_aec_env(**kwargs):
    from pettingzoo.utils.conversions import parallel_to_aec

    return parallel_to_aec(make_parallel_env(**kwargs))


def make_gym_env(**kwargs):
    from netforge_rl.environment.gym_env import NetForgeSingleAgentEnv

    return NetForgeSingleAgentEnv(**kwargs)


def register_gymnasium() -> tuple[str, ...]:
    import gymnasium as gym

    registered = []
    for env_id in GYM_IDS:
        if env_id not in gym.envs.registry:
            gym.register(
                id=env_id,
                entry_point=make_gym_env,
                max_episode_steps=200,
                kwargs={'scenario_type': 'ransomware', 'controlled_agent': 'blue_dmz'},
                disable_env_checker=True,
            )
        registered.append(env_id)
    return tuple(registered)


def register_pettingzoo() -> tuple[str, ...]:
    from pettingzoo import aec_registry, parallel_registry, register

    registered = []
    for env_id in PZ_IDS:
        if env_id not in parallel_registry:
            register(
                'parallel',
                env_id,
                make_parallel_env,
                max_cycles=200,
                kwargs={'scenario_type': 'ransomware', 'max_ticks': 200},
            )
        if env_id not in aec_registry:
            register(
                'aec',
                env_id,
                make_aec_env,
                max_cycles=200,
                kwargs={'scenario_type': 'ransomware', 'max_ticks': 200},
            )
        registered.append(env_id)
    return tuple(registered)


def register_envs() -> dict[str, tuple[str, ...]]:
    return {
        'gymnasium': register_gymnasium(),
        'pettingzoo': register_pettingzoo(),
    }
