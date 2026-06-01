"""JaxMARL-shape adapter tests."""

import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

from netforge_rl.backends.jax import VectorEnvSpec
from netforge_rl.bridges.jaxmarl import (
    DEFAULT_AGENTS,
    JaxMARLEnv,
    random_action_dict,
)


def _env(batch_size: int = 4) -> JaxMARLEnv:
    spec = VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3)
    return JaxMARLEnv(spec=spec, batch_size=batch_size)


@pytest.mark.fast
def test_reset_returns_per_agent_obs_dict() -> None:
    env = _env(batch_size=2)
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert set(obs.keys()) == set(DEFAULT_AGENTS)
    for v in obs.values():
        assert v.shape[0] == 2
    assert state.hosts.status.shape == (2, 100)


@pytest.mark.fast
def test_step_returns_jaxmarl_tuple() -> None:
    env = _env(batch_size=2)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    actions = random_action_dict(env, key)
    obs, new_state, reward, done, info = env.step(key, state, actions)

    assert set(reward.keys()) == set(DEFAULT_AGENTS)
    assert set(done.keys()) == set(DEFAULT_AGENTS)
    assert set(info.keys()) == set(DEFAULT_AGENTS)
    for v in reward.values():
        assert v.shape == (2,)
    assert new_state.current_tick.shape == (2,)


@pytest.mark.fast
def test_step_advances_tick() -> None:
    env = _env(batch_size=1)
    key = jax.random.PRNGKey(7)
    _, state = env.reset(key)
    before = int(state.current_tick[0])
    _, new_state, _, _, _ = env.step(key, state, random_action_dict(env, key))
    assert int(new_state.current_tick[0]) == before + 1


@pytest.mark.fast
def test_step_is_jit_compatible() -> None:
    env = _env(batch_size=2)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    actions = random_action_dict(env, key)

    jitted = jax.jit(env.step)
    obs, new_state, reward, done, info = jitted(key, state, actions)
    new_state.hosts.status.block_until_ready()
    assert new_state.hosts.status.shape == (2, 100)
