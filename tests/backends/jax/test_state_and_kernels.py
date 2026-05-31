"""JAX backend tests: PyTree registration, conversion round-trips, kernel
correctness vs the functional core, jit/vmap safety, and shape contracts.

Skipped automatically if JAX isn't importable.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

import numpy as np

from netforge_rl.backends.jax import (
    JaxEnvState,
    apply_compromised_by_delta,
    apply_host_privilege_delta,
    apply_host_status_delta,
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


AGENTS = (
    'red_operator',
    'blue_dmz',
    'blue_internal',
    'blue_restricted',
)


# ── PyTree registration ───────────────────────────────────────────────────


@pytest.mark.fast
def test_jax_envstate_is_a_registered_pytree(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    jstate = to_jax(snap)

    leaves, treedef = jax.tree_util.tree_flatten(jstate)
    # Every leaf should be a JAX array (no Python scalars / strings).
    for leaf in leaves:
        assert hasattr(leaf, 'shape'), f'non-array leaf {type(leaf).__name__}'

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, JaxEnvState)
    np.testing.assert_array_equal(rebuilt.hosts.status, jstate.hosts.status)


@pytest.mark.fast
def test_jax_envstate_works_under_vmap(global_state) -> None:
    """The PyTree must batch cleanly under vmap — this is the contract for
    the eventual 4096-env vectorization in Phase 2 slice 2."""
    snap = from_global_state(global_state, agent_ids=AGENTS)
    jstate = to_jax(snap)

    batched = jax.tree_util.tree_map(
        lambda x: jnp.stack([x, x, x]), jstate
    )

    @jax.vmap
    def identity(s: JaxEnvState) -> JaxEnvState:
        return s

    out = identity(batched)
    assert out.hosts.status.shape[0] == 3


# ── Round-trip ────────────────────────────────────────────────────────────


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


# ── Kernel correctness vs functional core ────────────────────────────────


@pytest.mark.fast
def test_status_kernel_matches_functional_core(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    ip = sorted(global_state.all_hosts.keys())[0]
    idx = snap.host_index(ip)

    cpu_after = apply_state_delta(snap, f'hosts/{ip}/status', 'isolated')
    jax_after = apply_host_status_delta(
        to_jax(snap),
        jnp.asarray(idx),
        jnp.asarray(STATUS_CODES.index('isolated'), dtype=jnp.int8),
    )

    np.testing.assert_array_equal(
        np.asarray(jax_after.hosts.status), cpu_after.hosts.status
    )


@pytest.mark.fast
def test_privilege_kernel_matches_functional_core(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    ip = sorted(global_state.all_hosts.keys())[0]
    idx = snap.host_index(ip)

    cpu_after = apply_state_delta(snap, f'hosts/{ip}/privilege', 'Root')
    jax_after = apply_host_privilege_delta(
        to_jax(snap),
        jnp.asarray(idx),
        jnp.asarray(PRIVILEGE_CODES.index('Root'), dtype=jnp.int8),
    )

    np.testing.assert_array_equal(
        np.asarray(jax_after.hosts.privilege), cpu_after.hosts.privilege
    )


@pytest.mark.fast
def test_compromised_by_kernel_matches_functional_core(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    ip = sorted(global_state.all_hosts.keys())[0]
    idx = snap.host_index(ip)

    cpu_after = apply_state_delta(snap, f'hosts/{ip}/compromised_by', 'red_operator')
    jax_after = apply_compromised_by_delta(
        to_jax(snap),
        jnp.asarray(idx),
        jnp.asarray(AGENTS.index('red_operator'), dtype=jnp.int8),
    )

    np.testing.assert_array_equal(
        np.asarray(jax_after.hosts.compromised_by_id),
        cpu_after.hosts.compromised_by_id,
    )


# ── jit safety ────────────────────────────────────────────────────────────


@pytest.mark.fast
def test_status_kernel_is_jit_safe(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    jstate = to_jax(snap)
    jitted = jax.jit(apply_host_status_delta)
    out = jitted(
        jstate,
        jnp.asarray(0),
        jnp.asarray(STATUS_CODES.index('isolated'), dtype=jnp.int8),
    )
    assert int(out.hosts.status[0]) == STATUS_CODES.index('isolated')


# ── Conflict resolution ──────────────────────────────────────────────────


@pytest.mark.fast
def test_resolve_conflicts_mask_no_collision() -> None:
    # 2 red, 2 blue; Red 0 targets host 0, Blue 0 defends host 1 -> no collision.
    red_targets = jnp.array([[True, False, False],
                             [False, True, False]], dtype=bool)
    blue_targets = jnp.array([[False, False, True],
                              [False, False, False]], dtype=bool)
    red_ok = jnp.array([True, True], dtype=bool)
    blue_ok = jnp.array([True, True], dtype=bool)

    surviving = resolve_conflicts_mask(red_targets, blue_targets, red_ok, blue_ok)
    np.testing.assert_array_equal(surviving, jnp.array([True, True]))


@pytest.mark.fast
def test_resolve_conflicts_mask_nullifies_on_collision() -> None:
    red_targets = jnp.array([[True, False, False],
                             [False, True, False]], dtype=bool)
    # Blue 0 defends host 0 (collides with Red 0); host 2 is irrelevant.
    blue_targets = jnp.array([[True, False, True],
                              [False, False, False]], dtype=bool)
    red_ok = jnp.array([True, True], dtype=bool)
    blue_ok = jnp.array([True, True], dtype=bool)

    surviving = resolve_conflicts_mask(red_targets, blue_targets, red_ok, blue_ok)
    np.testing.assert_array_equal(surviving, jnp.array([False, True]))


@pytest.mark.fast
def test_resolve_conflicts_mask_failed_blue_does_not_defend() -> None:
    red_targets = jnp.array([[True, False]], dtype=bool)
    blue_targets = jnp.array([[True, False]], dtype=bool)
    red_ok = jnp.array([True], dtype=bool)
    blue_ok = jnp.array([False], dtype=bool)  # Blue failed -> no defense
    surviving = resolve_conflicts_mask(red_targets, blue_targets, red_ok, blue_ok)
    np.testing.assert_array_equal(surviving, jnp.array([True]))


@pytest.mark.fast
def test_resolve_conflicts_mask_vmaps_over_envs() -> None:
    """Proof-of-life that the kernel batches under vmap — the headline
    Phase 2 promise of 'thousands of envs on one GPU' depends on this."""
    n_envs = 8
    red_targets = jnp.tile(
        jnp.array([[True, False, False]], dtype=bool)[None, :, :],
        (n_envs, 1, 1),
    )
    blue_targets = jnp.tile(
        jnp.array([[True, False, False]], dtype=bool)[None, :, :],
        (n_envs, 1, 1),
    )
    red_ok = jnp.ones((n_envs, 1), dtype=bool)
    blue_ok = jnp.ones((n_envs, 1), dtype=bool)

    batched = jax.vmap(resolve_conflicts_mask)(
        red_targets, blue_targets, red_ok, blue_ok
    )
    assert batched.shape == (n_envs, 1)
    # All envs collide -> all Red nullified.
    np.testing.assert_array_equal(batched, jnp.zeros((n_envs, 1), dtype=bool))
