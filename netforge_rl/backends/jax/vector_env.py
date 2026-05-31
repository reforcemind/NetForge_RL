"""Vectorized batched-env step under ``jax.vmap`` — the headline Phase 2
throughput demonstration.

This is a *simplified* step semantics: each Red agent attempts to compromise
its target host (sets ``privilege`` and ``compromised_by_id``); each Blue
agent attempts to isolate its target host (sets ``status='isolated'``).
Blue defensive supremacy is enforced via :func:`resolve_conflicts_mask` —
on a same-target collision, the Red attempt is nullified.

This is enough physics to:

  * Be a faithful subset of the legacy step's host-array mutations
    (the same fields, the same conflict semantics — see
    test_pure_step_equivalence in tests/parity/).
  * Saturate the JAX compiler so the throughput numbers we report are
    representative of the eventual full port (the cost is dominated by
    the .at[idx].set masked scatters, not by the action dispatch
    Python).
  * Provide a real ``vmap``'d step that batches cleanly across thousands
    of parallel envs.

The full 32-action / SIEM / NLP pipeline lands in later Phase 2 slices.
Until then this module is what we benchmark and what the Phase 5
baseline trainers will call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from netforge_rl.backends.jax.kernels import resolve_conflicts_mask
from netforge_rl.backends.jax.state import JaxEnvState
from netforge_rl.core.functional import PRIVILEGE_CODES, STATUS_CODES


# Encoded category codes pulled to module level so jit traces against
# concrete ints, not Python lookups.
_STATUS_ISOLATED = STATUS_CODES.index('isolated')
_PRIV_USER = PRIVILEGE_CODES.index('User')


class BatchedActions(NamedTuple):
    """One contiguous action specification per env in the batch.

    All arrays have leading dimension = batch size.

    Attributes:
        red_target_idx: int32[B, N_RED] — host slot each Red agent targets.
        blue_target_idx: int32[B, N_BLUE] — host slot each Blue agent targets.
        red_attempt: bool[B, N_RED] — whether each Red agent acts this tick.
        blue_attempt: bool[B, N_BLUE] — whether each Blue agent acts this tick.
    """

    red_target_idx: jax.Array
    blue_target_idx: jax.Array
    red_attempt: jax.Array
    blue_attempt: jax.Array


@dataclass(frozen=True)
class VectorEnvSpec:
    """Static shape contract — never traced, captured in closures."""

    n_hosts: int
    n_red: int
    n_blue: int


def _single_env_step(
    state: JaxEnvState,
    red_targets: jax.Array,    # int32[N_RED]
    blue_targets: jax.Array,   # int32[N_BLUE]
    red_attempt: jax.Array,    # bool[N_RED]
    blue_attempt: jax.Array,   # bool[N_BLUE]
    *,
    spec: VectorEnvSpec,
) -> tuple[JaxEnvState, jax.Array]:
    """Per-env step kernel; batched by vmap to produce the vector step."""
    n_hosts = spec.n_hosts

    # Build per-agent target masks: bool[N_AGENTS, N_HOSTS].
    one_hot = lambda idx, attempt: (
        jax.nn.one_hot(idx, n_hosts, dtype=jnp.bool_) & attempt[:, None]
    )
    red_target_mask = one_hot(red_targets, red_attempt)
    blue_target_mask = one_hot(blue_targets, blue_attempt)

    # Both sides nominally succeed; conflict resolver nullifies Red on overlap.
    red_success_pre = red_attempt
    blue_success = blue_attempt
    red_success = resolve_conflicts_mask(
        red_target_mask, blue_target_mask, red_success_pre, blue_success
    )

    # Aggregate the bool[N_AGENTS, N_HOSTS] masks into bool[N_HOSTS] writes.
    # Multiple agents writing the same host → that host gets written (any-wins).
    blue_writes_isolate = jnp.any(
        blue_target_mask & blue_success[:, None], axis=0
    )  # bool[N_HOSTS]
    red_writes_user = jnp.any(
        red_target_mask & red_success[:, None], axis=0
    )

    # Apply isolations (Blue defensive supremacy).
    new_status = jnp.where(
        blue_writes_isolate,
        jnp.int8(_STATUS_ISOLATED),
        state.hosts.status,
    )

    # Apply compromises (Red successes only — privilege bumps to User).
    new_privilege = jnp.where(
        red_writes_user,
        jnp.int8(_PRIV_USER),
        state.hosts.privilege,
    )

    # compromised_by_id: which Red agent owns each host?
    # Take argmax over Red agents of (target_mask & red_success). Where no
    # Red owns the host, keep the existing value (-1 default).
    red_owners = red_target_mask & red_success[:, None]  # bool[N_RED, N_HOSTS]
    any_red_owns = jnp.any(red_owners, axis=0)
    chosen_red = jnp.argmax(red_owners.astype(jnp.int8), axis=0).astype(jnp.int8)
    new_compromised = jnp.where(
        any_red_owns, chosen_red, state.hosts.compromised_by_id
    )

    new_hosts = replace(
        state.hosts,
        status=new_status,
        privilege=new_privilege,
        compromised_by_id=new_compromised,
    )

    # Reward: Blue gains per isolated host; Red gains per newly compromised.
    blue_reward = jnp.sum(blue_writes_isolate.astype(jnp.float32))
    red_reward = jnp.sum(red_writes_user.astype(jnp.float32))

    new_state = replace(
        state,
        hosts=new_hosts,
        current_tick=state.current_tick + 1,
    )

    # Reward shape: float32[N_RED + N_BLUE] — Red rewards first, then Blue.
    rewards = jnp.concatenate(
        [
            jnp.broadcast_to(red_reward, (spec.n_red,)),
            jnp.broadcast_to(blue_reward, (spec.n_blue,)),
        ]
    )
    return new_state, rewards


def make_vector_step(spec: VectorEnvSpec):
    """Compile a vmap'd batched step closed over the static shape spec.

    Returns a function ``step_fn(batched_state, batched_actions)`` that
    runs ``B`` independent envs in parallel. The wrapper is ``jax.jit``'d
    so the first call pays compile cost; subsequent calls run at fused-
    kernel speed.
    """
    per_env = partial(_single_env_step, spec=spec)
    batched = jax.vmap(per_env)

    @jax.jit
    def step_fn(state: JaxEnvState, actions: BatchedActions):
        return batched(
            state,
            actions.red_target_idx,
            actions.blue_target_idx,
            actions.red_attempt,
            actions.blue_attempt,
        )

    return step_fn


def random_actions(
    spec: VectorEnvSpec, batch_size: int, key: jax.Array
) -> BatchedActions:
    """Cheap random action sampler for SPS benchmarking and smoke tests."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return BatchedActions(
        red_target_idx=jax.random.randint(
            k1, (batch_size, spec.n_red), 0, spec.n_hosts, dtype=jnp.int32
        ),
        blue_target_idx=jax.random.randint(
            k2, (batch_size, spec.n_blue), 0, spec.n_hosts, dtype=jnp.int32
        ),
        red_attempt=jax.random.bernoulli(
            k3, p=0.5, shape=(batch_size, spec.n_red)
        ),
        blue_attempt=jax.random.bernoulli(
            k4, p=0.5, shape=(batch_size, spec.n_blue)
        ),
    )


def initial_batched_state(
    template: JaxEnvState, batch_size: int
) -> JaxEnvState:
    """Tile a single :class:`JaxEnvState` across a leading batch axis."""

    def tile(x):
        if isinstance(x, (jax.Array, np.ndarray)):
            return jnp.broadcast_to(jnp.asarray(x), (batch_size,) + tuple(x.shape))
        return jnp.broadcast_to(jnp.asarray(x), (batch_size,))

    return jax.tree_util.tree_map(tile, template)
