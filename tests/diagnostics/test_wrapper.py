import numpy as np
from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.diagnostics.wrapper import DiagnosticsWrapper


def test_diagnostics_wrapper():
    env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 10})
    wrapped_env = DiagnosticsWrapper(env)
    obs, infos = wrapped_env.reset(seed=42)
    for agent in wrapped_env.agents:
        assert 'oracle_obs' in infos[agent]
        assert 'information_asymmetry' in infos[agent]
        oracle_vec = infos[agent]['oracle_obs']
        agent_vec = obs[agent]['obs']
        assert oracle_vec.shape == agent_vec.shape
        distance = infos[agent]['information_asymmetry']
        assert distance > 0.0
        manual_distance = float(np.linalg.norm(oracle_vec - agent_vec))
        assert np.isclose(distance, manual_distance)
    actions = {
        agent: wrapped_env.action_space(agent).sample() for agent in wrapped_env.agents
    }
    obs, rewards, terminations, truncations, infos = wrapped_env.step(actions)
    for agent in wrapped_env.agents:
        assert 'oracle_obs' in infos[agent]
        assert 'information_asymmetry' in infos[agent]
        assert infos[agent]['information_asymmetry'] > 0.0
