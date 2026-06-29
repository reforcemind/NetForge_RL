import gymnasium as gym
from netforge_rl.bridges.rllib_bridge import NetForgeRLlibEnv


def test_rllib_env_init():
    env = NetForgeRLlibEnv({'scenario_type': 'ransomware'})
    assert 'red_operator' in env.possible_agents
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert isinstance(env.action_space, gym.spaces.Dict)
    assert env.observation_spaces
    assert env.action_spaces


def test_rllib_env_reset():
    env = NetForgeRLlibEnv({'scenario_type': 'ransomware'})
    obs, info = env.reset()
    assert 'red_operator' in obs
    assert 'red_operator' in info


def test_rllib_env_step():
    env = NetForgeRLlibEnv({'scenario_type': 'ransomware'})
    obs, info = env.reset()
    actions = {a: env.action_spaces[a].sample() for a in env.agents}
    obs, rew, term, trunc, info = env.step(actions)

    assert '__all__' in term
    assert '__all__' in trunc
    assert not term['__all__']

    import unittest.mock

    with unittest.mock.patch.object(
        env._env, 'step', return_value=({}, {}, {}, {}, {})
    ):
        obs, rew, term, trunc, info = env.step({})
    assert term.get('__all__', False)
