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
    OS_LINUX,
    OS_WINDOWS,
    PRIVILEGE_CODES,
    STATUS_CODES,
)


_STATUS_ONLINE = STATUS_CODES.index('online')
_STATUS_ISOLATED = STATUS_CODES.index('isolated')
_STATUS_KERNEL_PANIC = STATUS_CODES.index('kernel_panic')
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
RED_EXFILTRATE = 8           # Root-gated; +exfiltrated_bytes per tick
RED_SHARE_INTEL = 9          # OR every Red row of knowledge_mask together
RED_DUMP_LSASS = 10          # Root-gated; OR host_tokens[target] -> agent_credentials
RED_PASS_THE_HASH = 11       # compromise gated on token-locality (target's required token)
RED_PASS_THE_TICKET = 12     # privesc gated on token-locality
RED_JUICY_POTATO = 13        # Windows-only privesc (os_family == WINDOWS)
RED_V4L2 = 14                # Linux-only privesc (os_family == LINUX)
RED_KILL_PROCESS = 15        # Root-gated; status -> kernel_panic

BLUE_ISOLATE = 0
BLUE_RESTORE = 1
BLUE_DEPLOY_DECOY = 2
BLUE_DEPLOY_HONEYTOKEN = 3
BLUE_REMOVE = 4
BLUE_SAT = 5
BLUE_MONITOR = 6
BLUE_MISINFORM = 7           # decoy -> Apache (planted fake service)
BLUE_CONFIGURE_ACL = 8       # edr_active -> True (endpoint monitoring on)
BLUE_ROTATE_KERBEROS = 9     # clear every Red row of agent_credentials

SAT_DROP = 0.1

_DECOY_APACHE = DECOY_CODES.index('Apache')
EXFIL_PER_HOST = 5.0  # bytes-units per Rooted host per Exfiltrate tick


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
    red_is_exfil = (red_action_type == RED_EXFILTRATE) & red_success
    red_is_share = (red_action_type == RED_SHARE_INTEL) & red_success
    red_is_lsass = (red_action_type == RED_DUMP_LSASS) & red_success
    red_is_pth = (red_action_type == RED_PASS_THE_HASH) & red_success
    red_is_ptt = (red_action_type == RED_PASS_THE_TICKET) & red_success
    red_is_juicy = (red_action_type == RED_JUICY_POTATO) & red_success
    red_is_v4l2 = (red_action_type == RED_V4L2) & red_success
    red_is_kill = (red_action_type == RED_KILL_PROCESS) & red_success

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
    blue_is_misinform = (blue_action_type == BLUE_MISINFORM) & blue_attempt
    blue_is_acl = (blue_action_type == BLUE_CONFIGURE_ACL) & blue_attempt
    blue_is_rotate = (blue_action_type == BLUE_ROTATE_KERBEROS) & blue_attempt

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

    # LSASS: target host token row OR'd into the dumping Red agent's row,
    # gated on Root on that host.
    lsass_attempts = red_is_lsass & jnp.take(host_is_root, red_targets)
    looted_per_agent = jnp.where(
        lsass_attempts[:, None],
        jnp.take(state.hosts.host_tokens, red_targets, axis=0),
        jnp.zeros_like(state.agent_credentials[: spec.n_red]),
    )
    new_red_creds = state.agent_credentials[: spec.n_red] | looted_per_agent

    # PassTheHash / PassTheTicket: token-LOCALITY required — the agent must
    # already hold every token the target host advertises. Fixes the audit
    # bug where holding ANY looted token unlocked the entire network.
    red_creds = state.agent_credentials[: spec.n_red]      # bool[N_RED, N_TOKEN]
    target_tokens = jnp.take(state.hosts.host_tokens, red_targets, axis=0)
    held = jnp.where(target_tokens, red_creds, True).all(axis=-1)
    target_has_req = target_tokens.any(axis=-1)
    red_token_ok = held & target_has_req                   # bool[N_RED]

    red_writes_user_pth = jnp.any(
        red_target_mask & (red_is_pth & red_token_ok)[:, None], axis=0
    )
    red_writes_root_ptt = jnp.any(
        red_target_mask & (red_is_ptt & red_token_ok)[:, None], axis=0
    )
    red_writes_user = red_writes_user | red_writes_user_pth
    red_writes_root = red_writes_root | red_writes_root_ptt

    # JuicyPotato / V4L2: OS-gated privesc. Same effect as PRIVESC (User->Root)
    # but conditioned on host.os_family at the target.
    target_os = jnp.take(state.hosts.os_family, red_targets)
    juicy_ok = red_is_juicy & (target_os == OS_WINDOWS)
    v4l2_ok = red_is_v4l2 & (target_os == OS_LINUX)
    red_writes_root_juicy = jnp.any(
        red_target_mask & juicy_ok[:, None], axis=0
    )
    red_writes_root_v4l2 = jnp.any(
        red_target_mask & v4l2_ok[:, None], axis=0
    )
    red_writes_root = red_writes_root | red_writes_root_juicy | red_writes_root_v4l2

    # KillProcess: Root-gated; flips status to kernel_panic.
    red_writes_kill = jnp.any(
        red_target_mask & red_is_kill[:, None], axis=0
    ) & host_is_root

    # RotateKerberos: any Blue rotate clears every Red row.
    any_blue_rotate = jnp.any(blue_is_rotate)
    new_blue_creds = state.agent_credentials[spec.n_red:]
    new_red_creds = jnp.where(
        any_blue_rotate, jnp.zeros_like(new_red_creds), new_red_creds
    )
    new_agent_credentials = jnp.concatenate([new_red_creds, new_blue_creds], axis=0)

    # Audit fix: only count a Blue isolate as state-changing if the host
    # wasn't already isolated. Same idea applied to Red impact + kinetic
    # below — see the reward block.
    already_isolated = state.hosts.status == jnp.int8(_STATUS_ISOLATED)
    new_blue_isolations = blue_writes_isolate & ~already_isolated

    new_status = state.hosts.status
    new_status = jnp.where(blue_writes_restore, jnp.int8(_STATUS_ONLINE), new_status)
    new_status = jnp.where(blue_writes_isolate, jnp.int8(_STATUS_ISOLATED), new_status)
    new_status = jnp.where(red_writes_kill, jnp.int8(_STATUS_KERNEL_PANIC), new_status)

    new_privilege = state.hosts.privilege
    new_privilege = jnp.where(red_writes_user, jnp.int8(_PRIV_USER), new_privilege)
    new_privilege = jnp.where(red_writes_root, jnp.int8(_PRIV_ROOT), new_privilege)
    new_privilege = jnp.where(blue_writes_restore, jnp.int8(_PRIV_NONE), new_privilege)
    new_privilege = jnp.where(blue_writes_remove, jnp.int8(_PRIV_NONE), new_privilege)

    red_owners = (
        red_target_mask & (red_is_compromise | (red_is_pth & red_token_ok))[:, None]
    )
    any_red_owns = jnp.any(red_owners, axis=0)
    chosen_red = jnp.argmax(red_owners.astype(jnp.int8), axis=0).astype(jnp.int8)
    new_compromised = jnp.where(any_red_owns, chosen_red, state.hosts.compromised_by_id)
    new_compromised = jnp.where(blue_writes_restore, jnp.int8(-1), new_compromised)

    blue_writes_misinform = jnp.any(
        blue_target_mask & blue_is_misinform[:, None], axis=0
    )
    blue_writes_acl = jnp.any(blue_target_mask & blue_is_acl[:, None], axis=0)
    new_decoy = jnp.where(blue_writes_decoy, jnp.int8(_DECOY_ACTIVE), state.hosts.decoy)
    new_decoy = jnp.where(blue_writes_misinform, jnp.int8(_DECOY_APACHE), new_decoy)
    new_edr = state.hosts.edr_active | blue_writes_acl
    new_honey = state.hosts.contains_honeytokens | blue_writes_honey

    # Audit fix: count impact only on state-changing applications.
    already_compromised = state.hosts.system_integrity == jnp.int8(_INTEGRITY_COMPROMISED)
    already_kinetic = state.hosts.system_integrity == jnp.int8(_INTEGRITY_KINETIC)
    new_red_impacts = red_writes_impact & ~already_compromised & ~already_kinetic
    new_red_kinetics = red_writes_kinetic & ~already_kinetic

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
    new_red_knowledge = state.knowledge_mask[: spec.n_red] | red_knowledge_writes
    new_blue_knowledge = state.knowledge_mask[spec.n_red:] | blue_knowledge_writes

    # ShareIntel: if any Red agent shares, broadcast the OR of every Red
    # row to every Red row.
    any_red_shared = jnp.any(red_is_share)
    red_union = jnp.any(new_red_knowledge, axis=0)
    shared_rows = jnp.broadcast_to(red_union[None, :], new_red_knowledge.shape)
    new_red_knowledge = jnp.where(any_red_shared, shared_rows, new_red_knowledge)

    new_knowledge = jnp.concatenate([new_red_knowledge, new_blue_knowledge], axis=0)

    new_hosts = replace(
        state.hosts,
        status=new_status,
        privilege=new_privilege,
        compromised_by_id=new_compromised,
        decoy=new_decoy,
        contains_honeytokens=new_honey,
        system_integrity=new_integrity,
        human_vulnerability=new_human_vuln,
        edr_active=new_edr,
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

    exfil_targets = (
        red_target_mask & red_is_exfil[:, None] & host_is_root[None, :]
    )
    exfil_bytes_this_tick = (
        EXFIL_PER_HOST * jnp.sum(jnp.any(exfil_targets, axis=0).astype(jnp.float32))
    )

    # Audit fix 1.2: gate the farmable rewards on actual state transitions.
    blue_reward = (
        jnp.sum(new_blue_isolations.astype(jnp.float32))
        + 2.0 * jnp.sum(blue_writes_restore.astype(jnp.float32))
        + 0.5 * jnp.sum(blue_writes_decoy.astype(jnp.float32))
        + 0.5 * jnp.sum(blue_writes_honey.astype(jnp.float32))
        + 1.5 * jnp.sum(blue_writes_remove.astype(jnp.float32))
        + 0.3 * jnp.sum(blue_writes_sat.astype(jnp.float32))
        + 0.2 * jnp.sum(blue_new_intel.astype(jnp.float32))
        + 0.4 * jnp.sum(blue_writes_misinform.astype(jnp.float32))
        + 0.7 * jnp.sum(blue_writes_acl.astype(jnp.float32))
        + 4.0 * jnp.sum(blue_is_rotate.astype(jnp.float32))
    )
    red_team_reward = (
        jnp.sum(red_writes_user.astype(jnp.float32))
        + 0.5 * n_cve_compromises.astype(jnp.float32)
        + 3.0 * jnp.sum(red_writes_root.astype(jnp.float32))
        + 10.0 * jnp.sum(new_red_impacts.astype(jnp.float32))
        + 10_000.0 * jnp.sum(new_red_kinetics.astype(jnp.float32))
        + 0.2 * jnp.sum(red_new_intel.astype(jnp.float32))
        + exfil_bytes_this_tick
        + 0.4 * jnp.sum(red_is_share.astype(jnp.float32))
        + 2.0 * jnp.sum(jnp.any(looted_per_agent, axis=-1).astype(jnp.float32))
        + 5.0 * jnp.sum(red_writes_kill.astype(jnp.float32))
    )
    red_trap_penalty = -5.0 * red_trapped.astype(jnp.float32)

    new_state = replace(
        state,
        hosts=new_hosts,
        current_tick=state.current_tick + 1,
        knowledge_mask=new_knowledge,
        exfiltrated_bytes=state.exfiltrated_bytes + exfil_bytes_this_tick,
        agent_credentials=new_agent_credentials,
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
            k5, (batch_size, spec.n_red), 0, 16, dtype=jnp.int8
        ),
        blue_action_type=jax.random.randint(
            k6, (batch_size, spec.n_blue), 0, 10, dtype=jnp.int8
        ),
    )


def initial_batched_state(template: JaxEnvState, batch_size: int) -> JaxEnvState:
    """Tile a single JaxEnvState across a leading batch axis."""

    def tile(x):
        if isinstance(x, (jax.Array, np.ndarray)):
            return jnp.broadcast_to(jnp.asarray(x), (batch_size,) + tuple(x.shape))
        return jnp.broadcast_to(jnp.asarray(x), (batch_size,))

    return jax.tree_util.tree_map(tile, template)
