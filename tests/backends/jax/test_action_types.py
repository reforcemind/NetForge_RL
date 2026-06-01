"""Tests for Phase 2 slice 3 — RED_PRIVESC + BLUE_RESTORE action types."""

import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

from netforge_rl.backends.jax import (
    BatchedActions,
    VectorEnvSpec,
    initial_batched_state,
    make_vector_step,
    to_jax,
)
from netforge_rl.backends.jax.vector_env import (
    BLUE_DEPLOY_DECOY,
    BLUE_ISOLATE,
    BLUE_RESTORE,
    RED_COMPROMISE,
    RED_PRIVESC,
)
from netforge_rl.core.functional import (
    DECOY_CODES,
    PRIVILEGE_CODES,
    STATUS_CODES,
    from_global_state,
)


AGENTS = ('red_operator', 'blue_dmz', 'blue_internal', 'blue_restricted')


def _state(global_state, batch: int = 1):
    snap = from_global_state(global_state, agent_ids=AGENTS)
    return initial_batched_state(to_jax(snap), batch_size=batch)


def _spec(n_red: int = 1, n_blue: int = 1) -> VectorEnvSpec:
    return VectorEnvSpec(n_hosts=100, n_red=n_red, n_blue=n_blue)


def _act(*, red_t, blue_t, red_a, blue_a, red_type, blue_type) -> BatchedActions:
    return BatchedActions(
        red_target_idx=jnp.asarray(red_t, dtype=jnp.int32),
        blue_target_idx=jnp.asarray(blue_t, dtype=jnp.int32),
        red_attempt=jnp.asarray(red_a, dtype=jnp.bool_),
        blue_attempt=jnp.asarray(blue_a, dtype=jnp.bool_),
        red_action_type=jnp.asarray(red_type, dtype=jnp.int8),
        blue_action_type=jnp.asarray(blue_type, dtype=jnp.int8),
    )


# ── Red privesc ──────────────────────────────────────────────────────────


@pytest.mark.fast
def test_privesc_promotes_user_to_root(global_state) -> None:
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    # 1. Compromise host 0 (User).
    state, _ = step(state, _act(
        red_t=[[0]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, 0]) == PRIVILEGE_CODES.index('User')

    # 2. Privesc on host 0 (User -> Root).
    state, _ = step(state, _act(
        red_t=[[0]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_PRIVESC]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, 0]) == PRIVILEGE_CODES.index('Root')


@pytest.mark.fast
def test_privesc_noop_on_uncompromised_host(global_state) -> None:
    """Privesc against a host with privilege=None must NOT promote it."""
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    before = int(state.hosts.privilege[0, 5])
    state, _ = step(state, _act(
        red_t=[[5]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_PRIVESC]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, 5]) == before


# ── Blue restore ────────────────────────────────────────────────────────


@pytest.mark.fast
def test_restore_wipes_priv_and_ownership(global_state) -> None:
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    # Red compromises host 3.
    state, _ = step(state, _act(
        red_t=[[3]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, 3]) == PRIVILEGE_CODES.index('User')
    assert int(state.hosts.compromised_by_id[0, 3]) == 0

    # Blue restores host 3.
    state, _ = step(state, _act(
        red_t=[[99]], blue_t=[[3]],
        red_a=[[False]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_RESTORE]],
    ))
    assert int(state.hosts.privilege[0, 3]) == PRIVILEGE_CODES.index('None')
    assert int(state.hosts.compromised_by_id[0, 3]) == -1
    assert int(state.hosts.status[0, 3]) == STATUS_CODES.index('online')


@pytest.mark.fast
def test_restore_beats_simultaneous_red_compromise(global_state) -> None:
    """Blue defensive supremacy: same target, same tick -> Red nullified."""
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    pre_priv = int(state.hosts.privilege[0, 7])
    state, _ = step(state, _act(
        red_t=[[7]], blue_t=[[7]],
        red_a=[[True]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_RESTORE]],
    ))
    # Red collided -> nullified; restore on a clean host is effectively a no-op
    # on privilege but flips status to 'online' (which it already was).
    assert int(state.hosts.privilege[0, 7]) == pre_priv


# ── Reward shaping ──────────────────────────────────────────────────────


@pytest.mark.fast
def test_privesc_reward_is_higher_than_compromise(global_state) -> None:
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    state, _ = step(state, _act(
        red_t=[[0]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_ISOLATE]],
    ))
    _, r_privesc = step(state, _act(
        red_t=[[0]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_PRIVESC]], blue_type=[[BLUE_ISOLATE]],
    ))
    # Red reward column 0 should be 3.0 (privesc bonus) > 1.0 (compromise).
    assert float(r_privesc[0, 0]) == pytest.approx(3.0)


# ── Blue deploy decoy ────────────────────────────────────────────────────


@pytest.mark.fast
def test_deploy_decoy_flips_decoy_field(global_state) -> None:
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    # Pick a host that's not already a decoy.
    idx = int(
        next(
            i for i in range(100)
            if int(state.hosts.decoy[0, i]) == DECOY_CODES.index('inactive')
        )
    )
    state, rewards = step(state, _act(
        red_t=[[99]], blue_t=[[idx]],
        red_a=[[False]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_DEPLOY_DECOY]],
    ))
    assert int(state.hosts.decoy[0, idx]) == DECOY_CODES.index('active')
    # Blue reward 0 gets the +0.5 decoy bonus.
    assert float(rewards[0, spec.n_red]) == pytest.approx(0.5)
