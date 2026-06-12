"""Vectorized batched-step tests for the JAX backend."""

from __future__ import annotations

import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

import numpy as np

from netforge_rl.backends.jax import (
    BatchedActions,
    VectorEnvSpec,
    initial_batched_state,
    make_vector_step,
    random_actions,
    to_jax,
)
from netforge_rl.core.functional import (
    PRIVILEGE_CODES,
    STATUS_CODES,
    from_global_state,
)


AGENTS = (
    'red_operator',
    'blue_dmz',
    'blue_internal',
    'blue_restricted',
)


def _spec(n_hosts: int = 100, n_red: int = 1, n_blue: int = 3) -> VectorEnvSpec:
    return VectorEnvSpec(n_hosts=n_hosts, n_red=n_red, n_blue=n_blue)


@pytest.mark.fast
def test_batched_state_has_leading_batch_axis(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    state = to_jax(snap)
    batched = initial_batched_state(state, batch_size=4)
    assert batched.hosts.status.shape == (4, 100)
    assert batched.current_tick.shape == (4,)


@pytest.mark.fast
def test_step_preserves_shape(global_state) -> None:
    spec = _spec()
    snap = from_global_state(global_state, agent_ids=AGENTS)
    state = initial_batched_state(to_jax(snap), batch_size=8)

    step = make_vector_step(spec)
    actions = random_actions(spec, batch_size=8, key=jax.random.PRNGKey(0))
    new_state, rewards = step(state, actions)

    assert new_state.hosts.status.shape == (8, 100)
    assert new_state.hosts.privilege.shape == (8, 100)
    assert rewards.shape == (8, spec.n_red + spec.n_blue)
    np.testing.assert_array_equal(new_state.current_tick, jnp.ones(8, dtype=jnp.int32))


@pytest.mark.fast
def test_red_compromises_uncontested_host(global_state) -> None:
    spec = _spec(n_red=1, n_blue=1)
    snap = from_global_state(global_state, agent_ids=AGENTS)
    state = initial_batched_state(to_jax(snap), batch_size=1)
    step = make_vector_step(spec)

    actions = BatchedActions(
        red_target_idx=jnp.array([[0]], dtype=jnp.int32),
        blue_target_idx=jnp.array([[50]], dtype=jnp.int32),  # different host
        red_attempt=jnp.array([[True]], dtype=jnp.bool_),
        blue_attempt=jnp.array([[True]], dtype=jnp.bool_),
    )
    new_state, _ = step(state, actions)
    assert int(new_state.hosts.privilege[0, 0]) == PRIVILEGE_CODES.index('User')
    assert int(new_state.hosts.compromised_by_id[0, 0]) == 0  # Red 0


@pytest.mark.fast
def test_blue_isolates_then_red_blocked_on_same_target(global_state) -> None:
    spec = _spec(n_red=1, n_blue=1)
    snap = from_global_state(global_state, agent_ids=AGENTS)
    state = initial_batched_state(to_jax(snap), batch_size=1)
    step = make_vector_step(spec)

    actions = BatchedActions(
        red_target_idx=jnp.array([[7]], dtype=jnp.int32),
        blue_target_idx=jnp.array([[7]], dtype=jnp.int32),  # same target
        red_attempt=jnp.array([[True]], dtype=jnp.bool_),
        blue_attempt=jnp.array([[True]], dtype=jnp.bool_),
    )
    new_state, rewards = step(state, actions)

    # Blue defended -> host isolated, Red privilege unchanged.
    assert int(new_state.hosts.status[0, 7]) == STATUS_CODES.index('isolated')
    assert int(new_state.hosts.privilege[0, 7]) == int(state.hosts.privilege[0, 7])
    # Red got no reward; Blue got +1.
    assert float(rewards[0, 0]) == 0.0  # Red
    assert float(rewards[0, 1]) == 1.0  # Blue 0


@pytest.mark.fast
def test_envs_are_independent_under_vmap(global_state) -> None:
    """Each env in the batch must evolve independently — basic vmap correctness."""
    spec = _spec(n_red=1, n_blue=1)
    snap = from_global_state(global_state, agent_ids=AGENTS)
    state = initial_batched_state(to_jax(snap), batch_size=3)
    step = make_vector_step(spec)

    # Env 0: Red targets host 1. Env 1: no-op. Env 2: Blue isolates host 2.
    actions = BatchedActions(
        red_target_idx=jnp.array([[1], [0], [0]], dtype=jnp.int32),
        blue_target_idx=jnp.array([[99], [99], [2]], dtype=jnp.int32),
        red_attempt=jnp.array([[True], [False], [False]], dtype=jnp.bool_),
        blue_attempt=jnp.array([[False], [False], [True]], dtype=jnp.bool_),
    )
    new_state, _ = step(state, actions)

    # Env 0: host 1 compromised.
    assert int(new_state.hosts.privilege[0, 1]) == PRIVILEGE_CODES.index('User')
    # Env 1: state unchanged.
    np.testing.assert_array_equal(
        new_state.hosts.privilege[1], state.hosts.privilege[1]
    )
    # Env 2: host 2 isolated.
    assert int(new_state.hosts.status[2, 2]) == STATUS_CODES.index('isolated')


@pytest.mark.fast
def test_jit_step_is_compilable_at_large_batch(global_state) -> None:
    """4096-env compile must succeed (the headline scale promised in the
    NeurIPS abstract). We don't measure throughput here — that's the
    benchmark harness — just compile + correctness of one step."""
    spec = _spec()
    snap = from_global_state(global_state, agent_ids=AGENTS)
    state = initial_batched_state(to_jax(snap), batch_size=4096)
    step = make_vector_step(spec)
    actions = random_actions(spec, batch_size=4096, key=jax.random.PRNGKey(7))

    new_state, rewards = step(state, actions)
    new_state.hosts.status.block_until_ready()  # ensure exec completed
    assert new_state.hosts.status.shape == (4096, 100)
    assert rewards.shape == (4096, spec.n_red + spec.n_blue)
