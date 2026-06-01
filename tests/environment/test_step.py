import pytest
import numpy as np
from netforge_rl.environment.parallel_env import NetForgeRLEnv


@pytest.fixture
def env_sim_local(env_config):
    env = NetForgeRLEnv(env_config)
    env.reset(seed=42)
    return env


@pytest.mark.fast
def test_env_step_interaction(env_sim_local):
    """Verify that stepping returns rewards and observations for all agents."""
    env_sim_local.reset(seed=42)
    actions = {
        agent: env_sim_local.action_space(agent).sample()
        for agent in env_sim_local.agents
    }
    obs, rewards, terms, truncs, infos = env_sim_local.step(actions)
    assert len(obs) > 0
    assert len(rewards) > 0
    for r in rewards.values():
        assert isinstance(r, (int, float, np.float32, np.float64))


@pytest.mark.fast
def test_env_episode_truncation(env_sim_local):
    """Verify that episode truncates after max_ticks.

    Uses a fixed [0, 0] action (no queued event with duration > 1) so the
    event-driven time jump can't push current_tick past max_ticks in one
    step. action_space.sample() would be order-dependent here (gymnasium's
    stateful np_random).
    """
    env_sim_local.max_ticks = 2
    env_sim_local.reset(seed=42)
    actions = {a: np.array([0, 0], dtype=np.int64) for a in env_sim_local.agents}
    obs, rewards, terms, truncs, _ = env_sim_local.step(actions)
    assert all(not t for t in truncs.values())
    actions = {a: np.array([0, 0], dtype=np.int64) for a in env_sim_local.agents}
    obs, rewards, terms, truncs, _ = env_sim_local.step(actions)
    assert all(t for t in truncs.values())


@pytest.mark.fast
def test_blue_siem_embedding_update(env_sim_local):
    """Verify that DMZ Blue receives a non-zero embedding once a DMZ log arrives.

    We don't assert that other-subnet embeddings are zero — log_background_noise
    fires every step and uses the global ``random`` module, so the state of the
    other-subnet buffers depends on test ordering. The DMZ signal is the only
    load-bearing claim here.
    """
    env_sim_local.reset(seed=42)

    # Inject a realistic log to ensure non-zero embedding
    fake_log = "<Event xmlns='...'><System><EventID>4624</EventID></System></Event>"
    env_sim_local.siem_logger._push_to_buffer(
        fake_log, '192.168.1.0/24', env_sim_local.global_state
    )

    actions = {a: env_sim_local.action_space(a).sample() for a in env_sim_local.agents}
    obs, _, _, _, _ = env_sim_local.step(actions)

    dmz_obs = obs.get('blue_dmz')
    assert dmz_obs is not None, 'blue_dmz missing from observations'
    assert not np.allclose(dmz_obs['siem_embedding'], 0.0), (
        'blue_dmz embedding is zero — DMZ signal not propagating'
    )

    # Red agents must always see zeros (fog-of-war).
    for agent in ('red_commander', 'red_operator'):
        if agent in obs:
            assert np.allclose(obs[agent]['siem_embedding'], 0.0), (
                f'Embedding for {agent} is non-zero'
            )
