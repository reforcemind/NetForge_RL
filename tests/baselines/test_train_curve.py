import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

from netforge_rl.baselines.jax_ppo import (
    PPOConfig,
    forward,
    init_mlp_params,
    ippo_train,
    load_params,
    save_params,
)


@pytest.mark.fast
def test_checkpoint_roundtrip(tmp_path):
    params = init_mlp_params(
        jax.random.PRNGKey(0), obs_dim=384, n_targets=100, n_action_types=20
    )
    path = tmp_path / 'ckpt.npz'
    save_params(params, str(path))
    loaded = load_params(str(path))

    assert set(loaded) == set(params)
    obs = jnp.ones((4, 384))
    (t1, ty1), v1 = forward(params, obs)
    (t2, ty2), v2 = forward(loaded, obs)
    assert np.allclose(np.asarray(t1), np.asarray(t2))
    assert np.allclose(np.asarray(v1), np.asarray(v2))


@pytest.mark.integration
def test_ippo_train_smoke_produces_curve():
    cfg = PPOConfig(
        total_iters=3, num_steps=16, batch_size=32, num_minibatches=4, n_hosts=100
    )
    out = ippo_train(cfg)
    assert len(out['reward_curve']) == 3
    assert len(out['losses']) == 3
    assert all(np.isfinite(x) for x in out['reward_curve'])
