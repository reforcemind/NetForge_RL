import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from netforge_rl.baselines.policies import HeuristicRedPolicy
from netforge_rl.environment import NetForgeSingleAgentEnv
from netforge_rl.nlp.log_encoder import EMBEDDING_DIM

FLAT_DIM = 256 + EMBEDDING_DIM


@pytest.mark.fast
def test_spaces_and_reset():
    env = NetForgeSingleAgentEnv('ransomware', max_ticks=30)
    obs, info = env.reset(seed=0)
    assert obs.shape == (FLAT_DIM,)
    assert env.observation_space.contains(obs)
    assert 'action_mask' in info and info['action_mask'].shape == (132,)


@pytest.mark.fast
def test_step_returns_gym_five_tuple():
    env = NetForgeSingleAgentEnv('ransomware', max_ticks=30)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (FLAT_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


@pytest.mark.fast
def test_same_seed_same_first_obs():
    a = NetForgeSingleAgentEnv('ransomware', max_ticks=30).reset(seed=5)[0]
    b = NetForgeSingleAgentEnv('ransomware', max_ticks=30).reset(seed=5)[0]
    assert np.array_equal(a, b)


@pytest.mark.integration
def test_passes_gymnasium_env_checker():
    env = NetForgeSingleAgentEnv(
        'ransomware',
        controlled_agent='blue_dmz',
        opponents={'red_operator': HeuristicRedPolicy(seed=0)},
        max_ticks=40,
    )
    check_env(env, skip_render_check=True)
