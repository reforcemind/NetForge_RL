from __future__ import annotations
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
import numpy as np
from netforge_rl.backends.jax import (
    JaxEnvState,
    resolve_conflicts_mask,
    to_jax,
    to_numpy,
)
from netforge_rl.core.functional import (
    PRIVILEGE_CODES,
    STATUS_CODES,
    apply_state_delta,
    from_global_state,
)

AGENTS = ('red_operator', 'blue_dmz', 'blue_internal', 'blue_restricted')


@pytest.mark.fast
def test_jax_envstate_is_a_registered_pytree(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    jstate = to_jax(snap)
    leaves, treedef = jax.tree_util.tree_flatten(jstate)
    for leaf in leaves:
        assert hasattr(leaf, 'shape'), f'non-array leaf {type(leaf).__name__}'
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, JaxEnvState)
    np.testing.assert_array_equal(rebuilt.hosts.status, jstate.hosts.status)


@pytest.mark.fast
def test_jax_envstate_works_under_vmap(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    jstate = to_jax(snap)
    batched = jax.tree_util.tree_map(lambda x: jnp.stack([x, x, x]), jstate)

    @jax.vmap
    def identity(s: JaxEnvState) -> JaxEnvState:
        return s

    out = identity(batched)
    assert out.hosts.status.shape[0] == 3


@pytest.mark.fast
def test_to_jax_to_numpy_round_trip(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    jstate = to_jax(snap)
    back = to_numpy(jstate, snap.meta, snap.agent_ids)
    np.testing.assert_array_equal(back.hosts.status, snap.hosts.status)
    np.testing.assert_array_equal(back.hosts.privilege, snap.hosts.privilege)
    np.testing.assert_array_equal(
        back.hosts.compromised_by_id, snap.hosts.compromised_by_id
    )
    np.testing.assert_allclose(back.hosts.cvss_score, snap.hosts.cvss_score)
    assert back.current_tick == snap.current_tick


@pytest.mark.fast
def test_resolve_conflicts_mask_no_collision() -> None:
    red_targets = jnp.array([[True, False, False], [False, True, False]], dtype=bool)
    blue_targets = jnp.array([[False, False, True], [False, False, False]], dtype=bool)
    red_ok = jnp.array([True, True], dtype=bool)
    blue_ok = jnp.array([True, True], dtype=bool)
    surviving = resolve_conflicts_mask(red_targets, blue_targets, red_ok, blue_ok)
    np.testing.assert_array_equal(surviving, jnp.array([True, True]))


@pytest.mark.fast
def test_resolve_conflicts_mask_nullifies_on_collision() -> None:
    red_targets = jnp.array([[True, False, False], [False, True, False]], dtype=bool)
    blue_targets = jnp.array([[True, False, True], [False, False, False]], dtype=bool)
    red_ok = jnp.array([True, True], dtype=bool)
    blue_ok = jnp.array([True, True], dtype=bool)
    surviving = resolve_conflicts_mask(red_targets, blue_targets, red_ok, blue_ok)
    np.testing.assert_array_equal(surviving, jnp.array([False, True]))


@pytest.mark.fast
def test_resolve_conflicts_mask_failed_blue_does_not_defend() -> None:
    red_targets = jnp.array([[True, False]], dtype=bool)
    blue_targets = jnp.array([[True, False]], dtype=bool)
    red_ok = jnp.array([True], dtype=bool)
    blue_ok = jnp.array([False], dtype=bool)
    surviving = resolve_conflicts_mask(red_targets, blue_targets, red_ok, blue_ok)
    np.testing.assert_array_equal(surviving, jnp.array([True]))


@pytest.mark.fast
def test_resolve_conflicts_mask_vmaps_over_envs() -> None:
    n_envs = 8
    red_targets = jnp.tile(
        jnp.array([[True, False, False]], dtype=bool)[None, :, :], (n_envs, 1, 1)
    )
    blue_targets = jnp.tile(
        jnp.array([[True, False, False]], dtype=bool)[None, :, :], (n_envs, 1, 1)
    )
    red_ok = jnp.ones((n_envs, 1), dtype=bool)
    blue_ok = jnp.ones((n_envs, 1), dtype=bool)
    batched = jax.vmap(resolve_conflicts_mask)(
        red_targets, blue_targets, red_ok, blue_ok
    )
    assert batched.shape == (n_envs, 1)
    np.testing.assert_array_equal(batched, jnp.zeros((n_envs, 1), dtype=bool))
