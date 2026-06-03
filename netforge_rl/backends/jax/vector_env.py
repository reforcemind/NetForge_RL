from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from netforge_rl.backends.jax.kernels import resolve_conflicts_mask
from netforge_rl.backends.jax.state import JaxEnvState
from netforge_rl.core.functional import (
    CVE_CODES,
    DECOY_CODES,
    INTEGRITY_CODES,
    PRIVILEGE_CODES,
    STATUS_CODES,
)


_STATUS_ONLINE = STATUS_CODES.index('online')
_STATUS_ISOLATED = STATUS_CODES.index('isolated')
_PRIV_NONE = PRIVILEGE_CODES.index('None')
_PRIV_USER = PRIVILEGE_CODES.index('User')
_PRIV_ROOT = PRIVILEGE_CODES.index('Root')
_DECOY_ACTIVE = DECOY_CODES.index('active')
_INTEGRITY_CLEAN = INTEGRITY_CODES.index('clean')
_INTEGRITY_COMPROMISED = INTEGRITY_CODES.index('compromised')
_INTEGRITY_KINETIC = INTEGRITY_CODES.index('kinetic_destruction')
_CVE_BLUEKEEP = CVE_CODES.index('CVE-2019-0708')
_CVE_ETERNALBLUE = CVE_CODES.index('MS17-010')
_CVE_HTTP_RFI = CVE_CODES.index('CVE-2021-44228')

RED_COMPROMISE = 0
RED_PRIVESC = 1
RED_IMPACT = 2
RED_KINETIC = 3
RED_EXPLOIT_BLUEKEEP = 4
RED_EXPLOIT_ETERNALBLUE = 5
RED_EXPLOIT_HTTP_RFI = 6
RED_RECON = 7

BLUE_ISOLATE = 0
BLUE_RESTORE = 1
BLUE_DEPLOY_DECOY = 2
BLUE_DEPLOY_HONEYTOKEN = 3
BLUE_REMOVE = 4
BLUE_SAT = 5
BLUE_MONITOR = 6

SAT_DROP = 0.1


class BatchedActions(NamedTuple):
    red_target_idx: jax.Array
    blue_target_idx: jax.Array
    red_attempt: jax.Array
    blue_attempt: jax.Array
    red_action_type: jax.Array | None = None
    blue_action_type: jax.Array | None = None


@dataclass(frozen=True)
class VectorEnvSpec:
    n_hosts: int
    n_red: int
    n_blue: int


def _single_env_step(
    state,
    red_targets,
    blue_targets,
    red_attempt,
    blue_attempt,
    red_action_type,
    blue_action_type,
    *,
    spec: VectorEnvSpec,
):
    """One environment tick — batched by vmap to produce the vectorized step."""
    n_hosts = spec.n_hosts

    def one_hot(idx, attempt):
        return jax.nn.one_hot(idx, n_hosts, dtype=jnp.bool_) & attempt[:, None]

    red_target_mask = one_hot(red_targets, red_attempt)
    blue_target_mask = one_hot(blue_targets, blue_attempt)

    red_success = resolve_conflicts_mask(
        red_target_mask, blue_target_mask, red_attempt, blue_attempt
    )

    red_is_compromise = (red_action_type == RED_COMPROMISE) & red_success
    red_is_privesc = (red_action_type == RED_PRIVESC) & red_success
    red_is_impact = (red_action_type == RED_IMPACT) & red_success
    red_is_kinetic = (red_action_type == RED_KINETIC) & red_success
    red_is_recon = (red_action_type == RED_RECON) & red_success

    def _cve_compromise(action_code, cve_idx):
        is_attempt = (red_action_type == action_code) & red_success
        target_has_cve = jnp.take(state.hosts.vuln_mask[:, cve_idx], red_targets)
        return is_attempt & target_has_cve

    red_is_bluekeep = _cve_compromise(RED_EXPLOIT_BLUEKEEP, _CVE_BLUEKEEP)
    red_is_eternalblue = _cve_compromise(RED_EXPLOIT_ETERNALBLUE, _CVE_ETERNALBLUE)
    red_is_http_rfi = _cve_compromise(RED_EXPLOIT_HTTP_RFI, _CVE_HTTP_RFI)
    red_is_compromise = (
        red_is_compromise | red_is_bluekeep | red_is_eternalblue | red_is_http_rfi
    )

    blue_is_isolate = (blue_action_type == BLUE_ISOLATE) & blue_attempt
    blue_is_restore = (blue_action_type == BLUE_RESTORE) & blue_attempt
    blue_is_decoy = (blue_action_type == BLUE_DEPLOY_DECOY) & blue_attempt
    blue_is_honey = (blue_action_type == BLUE_DEPLOY_HONEYTOKEN) & blue_attempt
    blue_is_remove = (blue_action_type == BLUE_REMOVE) & blue_attempt
    blue_is_sat = (blue_action_type == BLUE_SAT) & blue_attempt
    blue_is_monitor = (blue_action_type == BLUE_MONITOR) & blue_attempt

    blue_writes_isolate = jnp.any(blue_target_mask & blue_is_isolate[:, None], axis=0)
    blue_writes_restore = jnp.any(blue_target_mask & blue_is_restore[:, None], axis=0)
    blue_writes_decoy = jnp.any(blue_target_mask & blue_is_decoy[:, None], axis=0)
    blue_writes_honey = jnp.any(blue_target_mask & blue_is_honey[:, None], axis=0)
    blue_writes_remove = jnp.any(blue_target_mask & blue_is_remove[:, None], axis=0)
    blue_writes_sat = jnp.any(blue_target_mask & blue_is_sat[:, None], axis=0)
    red_writes_user = jnp.any(red_target_mask & red_is_compromise[:, None], axis=0)
    red_writes_root = jnp.any(red_target_mask & red_is_privesc[:, None], axis=0)
    red_writes_impact = jnp.any(red_target_mask & red_is_impact[:, None], axis=0)
    red_writes_kinetic = jnp.any(red_target_mask & red_is_kinetic[:, None], axis=0)

    # Gating: privesc/impact/kinetic only succeed where the host is owned.
    host_already_owned = state.hosts.privilege >= jnp.int8(_PRIV_USER)
    host_is_root = state.hosts.privilege == jnp.int8(_PRIV_ROOT)
    red_writes_root = red_writes_root & host_already_owned
    red_writes_impact = red_writes_impact & host_is_root
    red_writes_kinetic = red_writes_kinetic & host_is_root

    new_status = state.hosts.status
    new_status = jnp.where(blue_writes_restore, jnp.int8(_STATUS_ONLINE), new_status)
    new_status = jnp.where(blue_writes_isolate, jnp.int8(_STATUS_ISOLATED), new_status)

    new_privilege = state.hosts.privilege
    new_privilege = jnp.where(red_writes_user, jnp.int8(_PRIV_USER), new_privilege)
    new_privilege = jnp.where(red_writes_root, jnp.int8(_PRIV_ROOT), new_privilege)
    new_privilege = jnp.where(blue_writes_restore, jnp.int8(_PRIV_NONE), new_privilege)
    new_privilege = jnp.where(blue_writes_remove, jnp.int8(_PRIV_NONE), new_privilege)

    red_owners = red_target_mask & red_is_compromise[:, None]
    any_red_owns = jnp.any(red_owners, axis=0)
    chosen_red = jnp.argmax(red_owners.astype(jnp.int8), axis=0).astype(jnp.int8)
    new_compromised = jnp.where(any_red_owns, chosen_red, state.hosts.compromised_by_id)
    new_compromised = jnp.where(blue_writes_restore, jnp.int8(-1), new_compromised)

    new_decoy = jnp.where(blue_writes_decoy, jnp.int8(_DECOY_ACTIVE), state.hosts.decoy)
    new_honey = state.hosts.contains_honeytokens | blue_writes_honey

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

    new_human_vuln = jnp.where(
        blue_writes_sat,
        jnp.maximum(state.hosts.human_vulnerability - SAT_DROP, 0.0),
        state.hosts.human_vulnerability,
    )

    red_knowledge_writes = red_target_mask & red_is_recon[:, None]
    blue_knowledge_writes = blue_target_mask & blue_is_monitor[:, None]
    new_knowledge = jnp.concatenate(
        [
            state.knowledge_mask[: spec.n_red] | red_knowledge_writes,
            state.knowledge_mask[spec.n_red:] | blue_knowledge_writes,
        ],
        axis=0,
    )

    new_hosts = replace(
        state.hosts,
        status=new_status,
        privilege=new_privilege,
        compromised_by_id=new_compromised,
        decoy=new_decoy,
        contains_honeytokens=new_honey,
        system_integrity=new_integrity,
        human_vulnerability=new_human_vuln,
    )

    red_trapped = jnp.any(
        red_target_mask
        & red_is_compromise[:, None]
        & state.hosts.contains_honeytokens[None, :],
        axis=1,
    )
    blue_new_intel = (~state.knowledge_mask[spec.n_red:]) & blue_knowledge_writes
    red_new_intel = (~state.knowledge_mask[: spec.n_red]) & red_knowledge_writes
    n_cve_compromises = (
        red_is_bluekeep.sum() + red_is_eternalblue.sum() + red_is_http_rfi.sum()
    )

    blue_reward = (
        jnp.sum(blue_writes_isolate.astype(jnp.float32))
        + 2.0 * jnp.sum(blue_writes_restore.astype(jnp.float32))
        + 0.5 * jnp.sum(blue_writes_decoy.astype(jnp.float32))
        + 0.5 * jnp.sum(blue_writes_honey.astype(jnp.float32))
        + 1.5 * jnp.sum(blue_writes_remove.astype(jnp.float32))
        + 0.3 * jnp.sum(blue_writes_sat.astype(jnp.float32))
        + 0.2 * jnp.sum(blue_new_intel.astype(jnp.float32))
    )
    red_team_reward = (
        jnp.sum(red_writes_user.astype(jnp.float32))
        + 0.5 * n_cve_compromises.astype(jnp.float32)
        + 3.0 * jnp.sum(red_writes_root.astype(jnp.float32))
        + 10.0 * jnp.sum(red_writes_impact.astype(jnp.float32))
        + 10_000.0 * jnp.sum(red_writes_kinetic.astype(jnp.float32))
        + 0.2 * jnp.sum(red_new_intel.astype(jnp.float32))
    )
    red_trap_penalty = -5.0 * red_trapped.astype(jnp.float32)

    new_state = replace(
        state,
        hosts=new_hosts,
        current_tick=state.current_tick + 1,
        knowledge_mask=new_knowledge,
    )

    red_rewards = jnp.broadcast_to(red_team_reward, (spec.n_red,)) + red_trap_penalty
    blue_rewards = jnp.broadcast_to(blue_reward, (spec.n_blue,))
    return new_state, jnp.concatenate([red_rewards, blue_rewards])


def _default_action_types(actions: BatchedActions, spec: VectorEnvSpec) -> BatchedActions:
    if actions.red_action_type is not None and actions.blue_action_type is not None:
        return actions
    batch = actions.red_target_idx.shape[0]
    return actions._replace(
        red_action_type=actions.red_action_type
        if actions.red_action_type is not None
        else jnp.zeros((batch, spec.n_red), dtype=jnp.int8),
        blue_action_type=actions.blue_action_type
        if actions.blue_action_type is not None
        else jnp.zeros((batch, spec.n_blue), dtype=jnp.int8),
    )


def make_vector_step(spec: VectorEnvSpec):
    """Return a jit'd vmap'd step closed over ``spec``."""
    per_env = partial(_single_env_step, spec=spec)
    batched = jax.vmap(per_env)

    @jax.jit
    def _step_impl(state, rt, bt, ra, ba, rat, bat):
        return batched(state, rt, bt, ra, ba, rat, bat)

    def step_fn(state, actions):
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


def random_actions(spec: VectorEnvSpec, batch_size: int, key: jax.Array) -> BatchedActions:
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    return BatchedActions(
        red_target_idx=jax.random.randint(
            k1, (batch_size, spec.n_red), 0, spec.n_hosts, dtype=jnp.int32
        ),
        blue_target_idx=jax.random.randint(
            k2, (batch_size, spec.n_blue), 0, spec.n_hosts, dtype=jnp.int32
        ),
        red_attempt=jax.random.bernoulli(k3, p=0.5, shape=(batch_size, spec.n_red)),
        blue_attempt=jax.random.bernoulli(k4, p=0.5, shape=(batch_size, spec.n_blue)),
        red_action_type=jax.random.randint(
            k5, (batch_size, spec.n_red), 0, 8, dtype=jnp.int8
        ),
        blue_action_type=jax.random.randint(
            k6, (batch_size, spec.n_blue), 0, 7, dtype=jnp.int8
        ),
    )


def initial_batched_state(template: JaxEnvState, batch_size: int) -> JaxEnvState:
    """Tile a single JaxEnvState across a leading batch axis."""

    def tile(x):
        if isinstance(x, (jax.Array, np.ndarray)):
            return jnp.broadcast_to(jnp.asarray(x), (batch_size,) + tuple(x.shape))
        return jnp.broadcast_to(jnp.asarray(x), (batch_size,))

    return jax.tree_util.tree_map(tile, template)
