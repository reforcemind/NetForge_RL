import numpy as np
import pytest

from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.environment.registry import (
    GYM_BLUE_ID,
    PZ_RANSOMWARE_ID,
    register_envs,
)
from netforge_rl.environment.spaces import FLAT_ACTION_MASK, CyberActionSpace


@pytest.mark.fast
def test_flat_action_mask_samples_on_multidiscrete():
    space = CyberActionSpace([32, 100])
    mask = np.zeros(FLAT_ACTION_MASK, dtype=np.int8)
    mask[0] = 1
    mask[32] = 1
    sample = space.sample(mask=mask)
    assert sample.shape == (2,)
    assert int(sample[0]) == 0
    assert int(sample[1]) == 0


@pytest.mark.fast
def test_env_action_space_accepts_obs_mask():
    env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 3})
    obs, _ = env.reset(seed=0)
    agent = env.agents[0]
    action = env.action_space(agent).sample(mask=obs[agent]['action_mask'])
    assert action.shape == (2,)


@pytest.mark.fast
def test_gymnasium_and_pettingzoo_register_and_make():
    register_envs()
    import gymnasium as gym
    from pettingzoo import make, parallel_registry

    assert GYM_BLUE_ID in gym.envs.registry
    assert PZ_RANSOMWARE_ID in parallel_registry
    gym_env = gym.make(GYM_BLUE_ID, max_ticks=4)
    _, info = gym_env.reset(seed=0)
    assert 'action_mask' in info
    gym_env.close()

    pz_env = make('parallel', PZ_RANSOMWARE_ID, max_ticks=4)
    pz_obs, _ = pz_env.reset(seed=0)
    assert 'blue_dmz' in pz_obs
    pz_env.close()
