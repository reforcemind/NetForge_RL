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
from netforge_rl.core.functional import DECOY_CODES, INTEGRITY_CODES

_STATUS_ONLINE = STATUS_CODES.index('online')
_STATUS_ISOLATED = STATUS_CODES.index('isolated')
_PRIV_NONE = PRIVILEGE_CODES.index('None')
_PRIV_USER = PRIVILEGE_CODES.index('User')
_PRIV_ROOT = PRIVILEGE_CODES.index('Root')
_DECOY_ACTIVE = DECOY_CODES.index('active')
_INTEGRITY_CLEAN = INTEGRITY_CODES.index('clean')
_INTEGRITY_COMPROMISED = INTEGRITY_CODES.index('compromised')
_INTEGRITY_KINETIC = INTEGRITY_CODES.index('kinetic_destruction')

# Action type encoding. Mirrors the legacy action_registry indexing for the
# *shared* subset that's currently implemented in the JAX backend.
RED_COMPROMISE = 0   # set privilege None -> User on target
RED_PRIVESC = 1      # User -> Root on already-owned target
RED_IMPACT = 2       # system_integrity -> compromised (ransomware-style)
RED_KINETIC = 3      # system_integrity -> kinetic_destruction (OT only)

BLUE_ISOLATE = 0     # status -> isolated
BLUE_RESTORE = 1     # privilege -> None, status -> online (heals the host)
BLUE_DEPLOY_DECOY = 2  # decoy -> active (proactive defense)
BLUE_DEPLOY_HONEYTOKEN = 3  # contains_honeytokens -> True (trap)


class BatchedActions(NamedTuple):
    """One contiguous action specification per env in the batch.

    All arrays have leading dimension = batch size.

    Attributes:
        red_target_idx: int32[B, N_RED] — host slot each Red agent targets.
        blue_target_idx: int32[B, N_BLUE] — host slot each Blue agent targets.
        red_attempt: bool[B, N_RED] — whether each Red agent acts this tick.
        blue_attempt: bool[B, N_BLUE] — whether each Blue agent acts this tick.
        red_action_type: int8[B, N_RED] — RED_COMPROMISE or RED_PRIVESC.
            Default-zero (back-compat) means COMPROMISE for every slot.
        blue_action_type: int8[B, N_BLUE] — BLUE_ISOLATE or BLUE_RESTORE.
    """

    red_target_idx: jax.Array
    blue_target_idx: jax.Array
    red_attempt: jax.Array
    blue_attempt: jax.Array
    red_action_type: jax.Array | None = None
    blue_action_type: jax.Array | None = None


@dataclass(frozen=True)
class VectorEnvSpec:
    """Static shape contract — never traced, captured in closures."""

    n_hosts: int
    n_red: int
    n_blue: int


def _single_env_step(
    state: JaxEnvState,
    red_targets: jax.Array,        # int32[N_RED]
    blue_targets: jax.Array,       # int32[N_BLUE]
    red_attempt: jax.Array,        # bool[N_RED]
    blue_attempt: jax.Array,       # bool[N_BLUE]
    red_action_type: jax.Array,    # int8[N_RED]; 0=compromise, 1=privesc
    blue_action_type: jax.Array,   # int8[N_BLUE]; 0=isolate, 1=restore
    *,
    spec: VectorEnvSpec,
) -> tuple[JaxEnvState, jax.Array]:
    """Per-env step kernel; batched by vmap to produce the vector step."""
    n_hosts = spec.n_hosts

    # Per-agent target masks: bool[N_AGENTS, N_HOSTS].
    one_hot = lambda idx, attempt: (
        jax.nn.one_hot(idx, n_hosts, dtype=jnp.bool_) & attempt[:, None]
    )
    red_target_mask = one_hot(red_targets, red_attempt)
    blue_target_mask = one_hot(blue_targets, blue_attempt)

    # Conflict resolution still applies — Blue defensive supremacy holds for
    # any same-target collision regardless of action type.
    red_success = resolve_conflicts_mask(
        red_target_mask, blue_target_mask, red_attempt, blue_attempt
    )

    # Split agent slots by chosen action type. Action type only matters when
    # the agent actually attempts (success masks already account for that).
    red_is_compromise = (red_action_type == RED_COMPROMISE) & red_success
    red_is_privesc = (red_action_type == RED_PRIVESC) & red_success
    red_is_impact = (red_action_type == RED_IMPACT) & red_success
    red_is_kinetic = (red_action_type == RED_KINETIC) & red_success
    blue_is_isolate = (blue_action_type == BLUE_ISOLATE) & blue_attempt
    blue_is_restore = (blue_action_type == BLUE_RESTORE) & blue_attempt
    blue_is_decoy = (blue_action_type == BLUE_DEPLOY_DECOY) & blue_attempt
    blue_is_honey = (blue_action_type == BLUE_DEPLOY_HONEYTOKEN) & blue_attempt

    # Aggregate to per-host write masks (any-wins across agents).
    blue_writes_isolate = jnp.any(
        blue_target_mask & blue_is_isolate[:, None], axis=0
    )  # bool[N_HOSTS]
    blue_writes_restore = jnp.any(
        blue_target_mask & blue_is_restore[:, None], axis=0
    )
    blue_writes_decoy = jnp.any(
        blue_target_mask & blue_is_decoy[:, None], axis=0
    )
    blue_writes_honey = jnp.any(
        blue_target_mask & blue_is_honey[:, None], axis=0
    )
    red_writes_user = jnp.any(
        red_target_mask & red_is_compromise[:, None], axis=0
    )
    red_writes_root = jnp.any(
        red_target_mask & red_is_privesc[:, None], axis=0
    )
    red_writes_impact = jnp.any(
        red_target_mask & red_is_impact[:, None], axis=0
    )
    red_writes_kinetic = jnp.any(
        red_target_mask & red_is_kinetic[:, None], axis=0
    )

    # Privesc only succeeds where the host is already at User+; reduces to
    # "no-op on a clean host", matching legacy ``required_prior_state``.
    host_already_owned = state.hosts.privilege >= jnp.int8(_PRIV_USER)
    red_writes_root = red_writes_root & host_already_owned

    # Impact + kinetic require Root on the target — matches legacy gating.
    host_is_root = state.hosts.privilege == jnp.int8(_PRIV_ROOT)
    red_writes_impact = red_writes_impact & host_is_root
    red_writes_kinetic = red_writes_kinetic & host_is_root

    # Status: Blue restore -> online; Blue isolate -> isolated; else keep.
    new_status = state.hosts.status
    new_status = jnp.where(blue_writes_restore, jnp.int8(_STATUS_ONLINE), new_status)
    new_status = jnp.where(blue_writes_isolate, jnp.int8(_STATUS_ISOLATED), new_status)

    # Privilege: Red privesc -> Root; Red compromise -> User; Blue restore ->
    # None. Order matters: restore wipes Red writes on the same host.
    new_privilege = state.hosts.privilege
    new_privilege = jnp.where(red_writes_user, jnp.int8(_PRIV_USER), new_privilege)
    new_privilege = jnp.where(red_writes_root, jnp.int8(_PRIV_ROOT), new_privilege)
    new_privilege = jnp.where(blue_writes_restore, jnp.int8(_PRIV_NONE), new_privilege)

    # compromised_by_id: pick the first successful Red owner per host.
    red_owners = red_target_mask & red_is_compromise[:, None]  # bool[N_RED, N_HOSTS]
    any_red_owns = jnp.any(red_owners, axis=0)
    chosen_red = jnp.argmax(red_owners.astype(jnp.int8), axis=0).astype(jnp.int8)
    new_compromised = jnp.where(
        any_red_owns, chosen_red, state.hosts.compromised_by_id
    )
    # Restore also clears ownership.
    new_compromised = jnp.where(
        blue_writes_restore, jnp.int8(-1), new_compromised
    )

    new_decoy = jnp.where(
        blue_writes_decoy, jnp.int8(_DECOY_ACTIVE), state.hosts.decoy
    )
    new_honey = state.hosts.contains_honeytokens | blue_writes_honey

    # system_integrity: kinetic > impact > existing. Blue restore wipes
    # back to 'clean'.
    new_integrity = state.hosts.system_integrity
    new_integrity = jnp.where(
        red_writes_impact, jnp.int8(_INTEGRITY_COMPROMISED), new_integrity
    )
    new_integrity = jnp.where(
        red_writes_kinetic, jnp.int8(_INTEGRITY_KINETIC), new_integrity
    )
    new_integrity = jnp.where(
        blue_writes_restore, jnp.int8(_INTEGRITY_CLEAN), new_integrity
    )

    new_hosts = replace(
        state.hosts,
        status=new_status,
        privilege=new_privilege,
        compromised_by_id=new_compromised,
        decoy=new_decoy,
        contains_honeytokens=new_honey,
        system_integrity=new_integrity,
    )

    # Reward: Blue +1 per isolate, +2 per successful restore (heal),
    # +0.5 per decoy (proactive but cheap), +0.5 per honeytoken;
    # Red +1 per fresh compromise, +3 per privesc, -5 for compromising a
    # honeytoken'd host (trap penalty).
    red_trapped = jnp.any(
        red_target_mask & red_is_compromise[:, None] & state.hosts.contains_honeytokens[None, :],
        axis=1,
    )  # bool[N_RED] — Red i hit a trap
    blue_reward = (
        jnp.sum(blue_writes_isolate.astype(jnp.float32))
        + 2.0 * jnp.sum(blue_writes_restore.astype(jnp.float32))
        + 0.5 * jnp.sum(blue_writes_decoy.astype(jnp.float32))
        + 0.5 * jnp.sum(blue_writes_honey.astype(jnp.float32))
    )
    red_team_reward = (
        jnp.sum(red_writes_user.astype(jnp.float32))
        + 3.0 * jnp.sum(red_writes_root.astype(jnp.float32))
        + 10.0 * jnp.sum(red_writes_impact.astype(jnp.float32))
        + 10_000.0 * jnp.sum(red_writes_kinetic.astype(jnp.float32))
    )
    # Trap penalty is per-Red-agent (not team-wide) so the trapped agent
    # gets the negative signal directly.
    red_trap_penalty = -5.0 * red_trapped.astype(jnp.float32)

    new_state = replace(
        state,
        hosts=new_hosts,
        current_tick=state.current_tick + 1,
    )

    # Reward shape: float32[N_RED + N_BLUE] — Red rewards first, then Blue.
    red_rewards = jnp.broadcast_to(red_team_reward, (spec.n_red,)) + red_trap_penalty
    blue_rewards = jnp.broadcast_to(blue_reward, (spec.n_blue,))
    rewards = jnp.concatenate([red_rewards, blue_rewards])
    return new_state, rewards


def _default_action_types(actions: BatchedActions, spec: VectorEnvSpec) -> BatchedActions:
    """Fill in zero action_type arrays when callers leave them unset."""
    if actions.red_action_type is None or actions.blue_action_type is None:
        batch = actions.red_target_idx.shape[0]
        return actions._replace(
            red_action_type=(
                actions.red_action_type
                if actions.red_action_type is not None
                else jnp.zeros((batch, spec.n_red), dtype=jnp.int8)
            ),
            blue_action_type=(
                actions.blue_action_type
                if actions.blue_action_type is not None
                else jnp.zeros((batch, spec.n_blue), dtype=jnp.int8)
            ),
        )
    return actions


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
    def _step_impl(state, rt, bt, ra, ba, rat, bat):
        return batched(state, rt, bt, ra, ba, rat, bat)

    def step_fn(state: JaxEnvState, actions: BatchedActions):
        actions = _default_action_types(actions, spec)
        return _step_impl(
            state,
            actions.red_target_idx,
            actions.blue_target_idx,
            actions.red_attempt,
            actions.blue_attempt,
            actions.red_action_type,
            actions.blue_action_type,
        )

    return step_fn


def random_actions(
    spec: VectorEnvSpec, batch_size: int, key: jax.Array
) -> BatchedActions:
    """Cheap random action sampler for SPS benchmarking and smoke tests."""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
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
        red_action_type=jax.random.randint(
            k5, (batch_size, spec.n_red), 0, 4, dtype=jnp.int8
        ),
        blue_action_type=jax.random.randint(
            k6, (batch_size, spec.n_blue), 0, 4, dtype=jnp.int8
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
