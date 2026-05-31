"""Vectorizable, jit-friendly state-mutation kernels for the JAX backend.

Each kernel takes a :class:`JaxEnvState` plus parameters (target host
index, encoded value) and returns a new :class:`JaxEnvState` with one
field updated. The kernels are written so they are safe under
``jax.jit`` and ``jax.vmap``:

  * Inputs are PyTrees of JAX arrays — no Python ints / strings in hot
    paths.
  * Indexed writes use ``arr.at[idx].set(value)`` (functional update)
    instead of in-place assignment.
  * No data-dependent Python branching on traced values — all conditional
    behavior happens through ``jnp.where`` / ``lax.cond``.

The conflict-resolution kernel works on per-step *masks* (one bool per
host) rather than on per-agent dicts. The legacy string-keyed delta
format is dispatched into masks by the host harness (see
:func:`netforge_rl.backends.jax.harness.deltas_to_masks` in a later
slice).
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from netforge_rl.backends.jax.state import JaxEnvState, JaxHostArrays


# ── Single-host attribute writes ──────────────────────────────────────────
#
# All three follow the same pattern: replace one slot of one host array.
# Kept as separate functions (rather than a generic _set_host_array(attr,
# idx, val)) because ``jax.jit`` can specialize each kernel separately,
# and the field name is a Python-level argument — passing it as a string
# would force a re-trace per call.


def apply_host_status_delta(
    state: JaxEnvState, host_idx: jax.Array, status_code: jax.Array
) -> JaxEnvState:
    new_status = state.hosts.status.at[host_idx].set(status_code)
    return replace(state, hosts=replace(state.hosts, status=new_status))


def apply_host_privilege_delta(
    state: JaxEnvState, host_idx: jax.Array, priv_code: jax.Array
) -> JaxEnvState:
    new_priv = state.hosts.privilege.at[host_idx].set(priv_code)
    return replace(state, hosts=replace(state.hosts, privilege=new_priv))


def apply_compromised_by_delta(
    state: JaxEnvState, host_idx: jax.Array, agent_code: jax.Array
) -> JaxEnvState:
    """Set ``compromised_by_id[host_idx] = agent_code``; pass -1 to clear."""
    new_arr = state.hosts.compromised_by_id.at[host_idx].set(agent_code)
    return replace(state, hosts=replace(state.hosts, compromised_by_id=new_arr))


# ── Conflict resolution as a vectorized mask op ───────────────────────────


def resolve_conflicts_mask(
    red_target_mask: jax.Array,   # bool[N_AGENTS, N_HOSTS], True = Red i targets host h
    blue_target_mask: jax.Array,  # bool[N_AGENTS, N_HOSTS], True = Blue i defends host h
    red_success: jax.Array,       # bool[N_AGENTS]
    blue_success: jax.Array,      # bool[N_AGENTS]
) -> jax.Array:
    """Pure mask form of :func:`netforge_rl.core.functional.resolve_conflicts`.

    Returns the post-resolution Red success vector. Red agent i is
    nullified iff any host they target overlaps any host that any
    successful Blue agent simultaneously defends.

    All operations are batchwise; the function is safe under
    ``jax.vmap`` over a leading env axis (just add a leading batch axis
    to every input).
    """
    # Defended hosts: any blue with success=True targeting that host.
    defended_hosts = jnp.any(
        blue_target_mask & blue_success[:, None], axis=0
    )  # bool[N_HOSTS]

    # Red collision: agent i's targets overlap the defended set.
    red_collisions = jnp.any(
        red_target_mask & defended_hosts[None, :], axis=1
    )  # bool[N_AGENTS]

    # Red i survives if it was successful AND uncollided.
    return red_success & ~red_collisions
