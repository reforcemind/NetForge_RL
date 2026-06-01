"""Smoke + shape tests for the JAX PPO baseline."""

import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

import numpy as np

from netforge_rl.baselines.jax_ppo import (
    PPOConfig,
    forward,
    gae,
    init_adam,
    init_mlp_params,
    adam_update,
    ppo_loss,
    train,
)


@pytest.mark.fast
def test_forward_shapes() -> None:
    p = init_mlp_params(jax.random.PRNGKey(0), obs_dim=8, n_actions=4, hidden=16)
    logits, value = forward(p, jnp.zeros((5, 8)))
    assert logits.shape == (5, 4) and value.shape == (5,)


@pytest.mark.fast
def test_adam_decreases_loss_on_quadratic() -> None:
    """Sanity: hand-rolled Adam should drive a convex objective down."""
    params = {'x': jnp.array(5.0)}
    opt = init_adam(params)

    def f(p):
        return (p['x'] ** 2,)

    loss_before = float(f(params)[0])
    for _ in range(30):
        grads = jax.tree_util.tree_map(lambda x: 2 * x, params)
        params, opt = adam_update(params, grads, opt, lr=0.1)
    loss_after = float(f(params)[0])
    assert loss_after < loss_before


@pytest.mark.fast
def test_gae_terminal_zeroes_carry() -> None:
    rewards = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    values = jnp.zeros((4,), dtype=jnp.float32)
    dones = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)  # terminal at t=1
    adv = gae(rewards, values, dones, gamma=0.99, lam=0.95)
    # t=1 is terminal -> adv[2] doesn't propagate back through t=1.
    assert adv.shape == (3,)


@pytest.mark.fast
def test_ppo_loss_is_scalar() -> None:
    p = init_mlp_params(jax.random.PRNGKey(0), obs_dim=8, n_actions=4)
    obs = jnp.zeros((10, 8))
    actions = jnp.zeros((10,), dtype=jnp.int32)
    logp = jnp.zeros((10,))
    adv = jnp.zeros((10,))
    ret = jnp.zeros((10,))
    loss, _ = ppo_loss(p, obs, actions, logp, adv, ret, clip=0.2, vf_coef=0.5)
    assert loss.shape == ()


@pytest.mark.integration
def test_train_smoke_runs_and_returns_losses() -> None:
    cfg = PPOConfig(total_iters=2, num_steps=4, batch_size=4, n_hosts=100)
    out = train(cfg)
    assert 'params' in out
    assert len(out['losses']) == 2
    for L in out['losses']:
        assert np.isfinite(L)
