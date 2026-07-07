import types

import jax
import jax.numpy as jnp
import pytest

from netforge_rl.backends.jax import VectorEnvSpec
from netforge_rl.backends.jax.vector_env import jax_siem_features
from netforge_rl.bridges.jaxmarl import JaxMARLEnv


def _hosts(compromised, privilege, honeytoken, decoy, edr):
    return types.SimpleNamespace(
        compromised_by_id=jnp.asarray(compromised),
        privilege=jnp.asarray(privilege),
        contains_honeytokens=jnp.asarray(honeytoken),
        decoy=jnp.asarray(decoy),
        edr_active=jnp.asarray(edr),
    )


@pytest.mark.fast
def test_alert_discriminates_compromise():
    hosts = _hosts(
        compromised=[-1, 0],  # host 1 is compromised
        privilege=[0, 1],
        honeytoken=[False, False],
        decoy=[0, 0],
        edr=[0, 0],
    )
    alert = jax_siem_features(hosts)
    assert float(alert[1]) > float(alert[0])
    assert float(alert[0]) == 0.0


@pytest.mark.fast
def test_alert_is_bounded():
    hosts = _hosts(
        compromised=[0, 0],
        privilege=[1, 1],
        honeytoken=[True, False],
        decoy=[1, 0],
        edr=[1, 0],
    )
    alert = jax_siem_features(hosts)
    assert float(alert.min()) >= 0.0 and float(alert.max()) <= 1.0


@pytest.mark.integration
def test_jit_compatible_on_real_state():
    spec = VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3)
    env = JaxMARLEnv(spec=spec, batch_size=4)
    _, state = env.reset(jax.random.PRNGKey(0))
    fn = jax.jit(lambda s: jax_siem_features(s.hosts))
    alert = fn(state)
    assert alert.shape == (4, spec.n_hosts)
    assert float(alert.min()) >= 0.0 and float(alert.max()) <= 1.0


@pytest.mark.integration
def test_telemetry_obs_widens_blue_only():
    spec = VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3)
    base = JaxMARLEnv(spec=spec, batch_size=4)
    tel = JaxMARLEnv(spec=spec, batch_size=4, telemetry_obs=True)
    key = jax.random.PRNGKey(0)
    ob, _ = base.reset(key)
    obt, _ = tel.reset(key)
    assert obt['blue_dmz'].shape[-1] == ob['blue_dmz'].shape[-1] + spec.n_hosts
    assert obt['red_operator'].shape == ob['red_operator'].shape
