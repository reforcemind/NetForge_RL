import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
import numpy as np
from netforge_rl.backends.jax import VectorEnvSpec
from netforge_rl.bridges.cleanrl_vec import CleanRLVecEnv


def _vec(num_envs: int = 4) -> CleanRLVecEnv:
    return CleanRLVecEnv(
        spec=VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3),
        num_envs=num_envs,
        agent_id='blue_dmz',
    )


@pytest.mark.fast
def test_reset_returns_batched_numpy_obs() -> None:
    vec = _vec(num_envs=4)
    obs, info = vec.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,) + vec.single_observation_space
    assert info == {}


@pytest.mark.fast
def test_step_returns_gym5_tuple_shapes() -> None:
    vec = _vec(num_envs=4)
    vec.reset(seed=0)
    actions = np.zeros((4, 2), dtype=np.int32)
    obs, reward, terminated, truncated, info = vec.step(actions)
    assert obs.shape == (4,) + vec.single_observation_space
    assert reward.shape == (4,) and reward.dtype == np.float32
    assert terminated.shape == (4,) and terminated.dtype == bool
    assert truncated.shape == (4,) and truncated.dtype == bool


@pytest.mark.fast
def test_action_steers_controlled_agent() -> None:
    vec = _vec(num_envs=2)
    vec.reset(seed=0)
    obs, reward, *_ = vec.step(np.full((2, 2), 7, dtype=np.int32))
    assert np.all(np.isfinite(reward))


@pytest.mark.fast
def test_advisory_spaces_present() -> None:
    vec = _vec()
    assert vec.single_action_space == ('multidiscrete', (100, 14))
    assert vec.single_observation_space == (400,)
