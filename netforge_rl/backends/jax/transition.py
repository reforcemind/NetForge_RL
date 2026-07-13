from dataclasses import replace

import jax
import jax.numpy as jnp

from netforge_rl.backends.jax.action_codes import (
    ACTION_DURATIONS_BLUE,
    ACTION_DURATIONS_RED,
    BLUE_ANALYZE,
    BLUE_CONFIGURE_ACL,
    BLUE_DEPLOY_DECOY,
    BLUE_DEPLOY_HONEYTOKEN,
    BLUE_ISOLATE,
    BLUE_MISINFORM_APACHE,
    BLUE_MISINFORM_SSHD,
    BLUE_MISINFORM_TOMCAT,
    BLUE_MONITOR,
    BLUE_REMOVE,
    BLUE_RESTORE,
    BLUE_RESTORE_FROM_BACKUP,
    BLUE_ROTATE_KERBEROS,
    BLUE_SAT,
    EXFIL_PER_HOST,
    RED_COMPROMISE,
    RED_DISCOVER_REMOTE_SYSTEMS,
    RED_DUMP_LSASS,
    RED_EXFILTRATE,
    RED_EXPLOIT_BLUEKEEP,
    RED_EXPLOIT_ETERNALBLUE,
    RED_EXPLOIT_HTTP_RFI,
    RED_IMPACT,
    RED_IP_FRAGMENTATION,
    RED_JUICY_POTATO,
    RED_KILL_PROCESS,
    RED_KINETIC,
    RED_NETWORK_SCAN,
    RED_PASS_THE_HASH,
    RED_PASS_THE_TICKET,
    RED_PRIVESC,
    RED_RECON,
    RED_SHARE_INTEL,
    RED_SPEARPHISHING,
    RED_V4L2,
    SAT_DROP,
)
from netforge_rl.backends.jax.kernels import resolve_conflicts_mask
from netforge_rl.backends.jax.rewards import StepEvents, compute_rewards
from netforge_rl.backends.jax.scenario_config import (
    SCENARIO_APT,
    SCENARIO_CLOUD,
    SCENARIO_IOT,
    SCENARIO_OT,
)
from netforge_rl.backends.jax.state_codes import (
    _CVE_BLUEKEEP,
    _CVE_ETERNALBLUE,
    _CVE_HTTP_RFI,
    _DECOY_ACTIVE,
    _DECOY_APACHE,
    _DECOY_SSHD,
    _DECOY_TOMCAT,
    _INTEGRITY_CLEAN,
    _INTEGRITY_COMPROMISED,
    _INTEGRITY_KINETIC,
    _PRIV_NONE,
    _PRIV_ROOT,
    _PRIV_USER,
    _STATUS_ISOLATED,
    _STATUS_KERNEL_PANIC,
    _STATUS_ONLINE,
)
from netforge_rl.core.functional import OS_LINUX, OS_WINDOWS


def _process_action_queue(
    state,
    red_targets,
    blue_targets,
    red_attempt,
    blue_attempt,
    red_action_type,
    blue_action_type,
    spec,
):
    """Enqueue newly-submitted actions and resolve whichever queued action
    matures this tick, mirroring the Python engine's timing:
    """
    n_red = spec.n_red
    n_streams = n_red + spec.n_blue
    full_q = (
        state.in_flight_actions
    )  # int32[N_AGENTS, 4]: (type, target, comp_tick, active)
    q = full_q[:n_streams]
    next_tick = state.current_tick + 1

    all_targets = jnp.concatenate([red_targets, blue_targets], axis=0).astype(jnp.int32)
    all_attempt = jnp.concatenate([red_attempt, blue_attempt], axis=0)
    all_action_type = jnp.concatenate(
        [red_action_type, blue_action_type], axis=0
    ).astype(jnp.int32)
    durations = jnp.concatenate(
        [
            ACTION_DURATIONS_RED[red_action_type.astype(jnp.int32)],
            ACTION_DURATIONS_BLUE[blue_action_type.astype(jnp.int32)],
        ],
        axis=0,
    )

    active = q[:, 3] == 1
    locked = active & (q[:, 2] > state.current_tick)
    can_submit = all_attempt & ~locked

    new_entry = jnp.stack(
        [
            all_action_type,
            all_targets,
            state.current_tick + durations,
            jnp.ones_like(all_action_type),
        ],
        axis=-1,
    )
    q = jnp.where(can_submit[:, None], new_entry, q)

    blue_q = q[n_red:]
    blue_maturing = (blue_q[:, 3] == 1) & (blue_q[:, 2] <= next_tick)
    blue_isolating = blue_maturing & (blue_q[:, 0] == BLUE_ISOLATE)
    isolated_mask = jnp.any(
        jax.nn.one_hot(blue_q[:, 1], spec.n_hosts, dtype=jnp.bool_)
        & blue_isolating[:, None],
        axis=0,
    )

    red_q = q[:n_red]
    red_active = red_q[:, 3] == 1
    red_cancelled = red_active & isolated_mask[red_q[:, 1]]
    red_q = red_q.at[:, 3].set(jnp.where(red_cancelled, 0, red_q[:, 3]))
    red_maturing = (red_q[:, 3] == 1) & (red_q[:, 2] <= next_tick)

    q = jnp.concatenate([red_q, blue_q], axis=0)
    maturing = jnp.concatenate([red_maturing, blue_maturing], axis=0)

    out_target = q[:, 1]
    out_action_type = q[:, 0].astype(jnp.int8)
    q_final = q.at[:, 3].set(jnp.where(maturing, 0, q[:, 3]))
    q_final = full_q.at[:n_streams].set(q_final)

    return (
        q_final,
        out_target[:n_red],
        out_target[n_red:],
        maturing[:n_red],
        maturing[n_red:],
        out_action_type[:n_red],
        out_action_type[n_red:],
    )


def single_env_step(
    state,
    red_targets,
    blue_targets,
    red_attempt,
    blue_attempt,
    red_action_type,
    blue_action_type,
    *,
    spec,
):
    """One vectorized environment tick: enqueue submitted actions, resolve
    whichever actions mature this tick, apply their effects, return
    (new_state, rewards)."""
    (
        new_queue,
        red_targets,
        blue_targets,
        red_attempt,
        blue_attempt,
        red_action_type,
        blue_action_type,
    ) = _process_action_queue(
        state,
        red_targets,
        blue_targets,
        red_attempt,
        blue_attempt,
        red_action_type,
        blue_action_type,
        spec,
    )

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
    red_is_recon = (
        (red_action_type == RED_RECON)
        | (red_action_type == RED_NETWORK_SCAN)
        | (red_action_type == RED_DISCOVER_REMOTE_SYSTEMS)
    ) & red_success
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
    red_is_ip_frag = (red_action_type == RED_IP_FRAGMENTATION) & red_success
    red_is_spearphishing = (red_action_type == RED_SPEARPHISHING) & red_success
    target_os_early = jnp.take(state.hosts.os_family, red_targets)
    spearphish_ok = red_is_spearphishing & (target_os_early == OS_WINDOWS)
    red_is_compromise = (
        red_is_compromise
        | red_is_bluekeep
        | red_is_eternalblue
        | red_is_http_rfi
        | red_is_ip_frag
        | spearphish_ok
    )

    blue_is_isolate = (blue_action_type == BLUE_ISOLATE) & blue_attempt
    blue_is_restore = (blue_action_type == BLUE_RESTORE) & blue_attempt
    blue_is_restore_backup = (
        blue_action_type == BLUE_RESTORE_FROM_BACKUP
    ) & blue_attempt
    blue_is_decoy = (blue_action_type == BLUE_DEPLOY_DECOY) & blue_attempt
    blue_is_honey = (blue_action_type == BLUE_DEPLOY_HONEYTOKEN) & blue_attempt
    blue_is_remove = (blue_action_type == BLUE_REMOVE) & blue_attempt
    blue_is_sat = (blue_action_type == BLUE_SAT) & blue_attempt
    blue_is_monitor = (blue_action_type == BLUE_MONITOR) & blue_attempt
    blue_is_analyze = (blue_action_type == BLUE_ANALYZE) & blue_attempt
    blue_is_misinform_apache = (
        blue_action_type == BLUE_MISINFORM_APACHE
    ) & blue_attempt
    blue_is_misinform_tomcat = (
        blue_action_type == BLUE_MISINFORM_TOMCAT
    ) & blue_attempt
    blue_is_misinform_sshd = (blue_action_type == BLUE_MISINFORM_SSHD) & blue_attempt
    blue_is_acl = (blue_action_type == BLUE_CONFIGURE_ACL) & blue_attempt
    blue_is_rotate = (blue_action_type == BLUE_ROTATE_KERBEROS) & blue_attempt

    blue_writes_isolate = jnp.any(blue_target_mask & blue_is_isolate[:, None], axis=0)
    blue_writes_restore = jnp.any(blue_target_mask & blue_is_restore[:, None], axis=0)
    blue_writes_restore_backup = jnp.any(
        blue_target_mask & blue_is_restore_backup[:, None], axis=0
    )
    blue_writes_any_restore = blue_writes_restore | blue_writes_restore_backup
    blue_writes_decoy = jnp.any(blue_target_mask & blue_is_decoy[:, None], axis=0)
    blue_writes_honey = jnp.any(blue_target_mask & blue_is_honey[:, None], axis=0)
    blue_writes_remove = jnp.any(blue_target_mask & blue_is_remove[:, None], axis=0)
    blue_writes_sat = jnp.any(blue_target_mask & blue_is_sat[:, None], axis=0)
    red_writes_user = jnp.any(red_target_mask & red_is_compromise[:, None], axis=0)
    red_writes_root = jnp.any(red_target_mask & red_is_privesc[:, None], axis=0)
    red_writes_impact = jnp.any(red_target_mask & red_is_impact[:, None], axis=0)
    red_writes_kinetic = jnp.any(red_target_mask & red_is_kinetic[:, None], axis=0)

    # Privilege/impact actions require host ownership
    host_already_owned = state.hosts.privilege >= jnp.int8(_PRIV_USER)
    host_is_root = state.hosts.privilege == jnp.int8(_PRIV_ROOT)
    red_writes_root = red_writes_root & host_already_owned
    red_writes_impact = red_writes_impact & host_is_root
    red_writes_kinetic = red_writes_kinetic & host_is_root

    # LSASS requires Root; ORs target tokens into agent credentials
    lsass_attempts = red_is_lsass & jnp.take(host_is_root, red_targets)
    looted_per_agent = jnp.where(
        lsass_attempts[:, None],
        jnp.take(state.hosts.host_tokens, red_targets, axis=0),
        jnp.zeros_like(state.agent_credentials[: spec.n_red]),
    )
    new_red_creds = state.agent_credentials[: spec.n_red] | looted_per_agent

    # PTH/PTT requires token locality
    red_creds = state.agent_credentials[: spec.n_red]
    target_tokens = jnp.take(state.hosts.host_tokens, red_targets, axis=0)
    held = jnp.where(target_tokens, red_creds, True).all(axis=-1)
    target_has_req = target_tokens.any(axis=-1)
    red_token_ok = held & target_has_req

    red_writes_user_pth = jnp.any(
        red_target_mask & (red_is_pth & red_token_ok)[:, None], axis=0
    )
    red_writes_root_ptt = jnp.any(
        red_target_mask & (red_is_ptt & red_token_ok)[:, None], axis=0
    )
    red_writes_user = red_writes_user | red_writes_user_pth
    red_writes_root = red_writes_root | red_writes_root_ptt

    # OS-gated privilege escalation
    target_os = jnp.take(state.hosts.os_family, red_targets)
    juicy_ok = red_is_juicy & (target_os == OS_WINDOWS)
    v4l2_ok = red_is_v4l2 & (target_os == OS_LINUX)
    red_writes_root_juicy = jnp.any(red_target_mask & juicy_ok[:, None], axis=0)
    red_writes_root_v4l2 = jnp.any(red_target_mask & v4l2_ok[:, None], axis=0)
    red_writes_root = red_writes_root | red_writes_root_juicy | red_writes_root_v4l2

    # KillProcess requires Root; sets kernel_panic
    red_writes_kill = (
        jnp.any(red_target_mask & red_is_kill[:, None], axis=0) & host_is_root
    )

    # RotateKerberos clears Red credentials
    any_blue_rotate = jnp.any(blue_is_rotate)
    new_blue_creds = state.agent_credentials[spec.n_red :]
    new_red_creds = jnp.where(
        any_blue_rotate, jnp.zeros_like(new_red_creds), new_red_creds
    )
    new_agent_credentials = jnp.concatenate([new_red_creds, new_blue_creds], axis=0)

    # Only count state-changing isolations
    already_isolated = state.hosts.status == jnp.int8(_STATUS_ISOLATED)
    new_blue_isolations = blue_writes_isolate & ~already_isolated

    new_status = state.hosts.status
    new_status = jnp.where(
        blue_writes_any_restore, jnp.int8(_STATUS_ONLINE), new_status
    )
    new_status = jnp.where(blue_writes_isolate, jnp.int8(_STATUS_ISOLATED), new_status)
    new_status = jnp.where(red_writes_kill, jnp.int8(_STATUS_KERNEL_PANIC), new_status)

    new_privilege = state.hosts.privilege
    new_privilege = jnp.where(red_writes_user, jnp.int8(_PRIV_USER), new_privilege)
    new_privilege = jnp.where(red_writes_root, jnp.int8(_PRIV_ROOT), new_privilege)
    new_privilege = jnp.where(
        blue_writes_any_restore, jnp.int8(_PRIV_NONE), new_privilege
    )
    new_privilege = jnp.where(blue_writes_remove, jnp.int8(_PRIV_NONE), new_privilege)

    red_owners = (
        red_target_mask & (red_is_compromise | (red_is_pth & red_token_ok))[:, None]
    )
    any_red_owns = jnp.any(red_owners, axis=0)
    chosen_red = jnp.argmax(red_owners.astype(jnp.int8), axis=0).astype(jnp.int8)
    new_compromised = jnp.where(any_red_owns, chosen_red, state.hosts.compromised_by_id)
    new_compromised = jnp.where(blue_writes_any_restore, jnp.int8(-1), new_compromised)

    blue_writes_misinform_apache = jnp.any(
        blue_target_mask & blue_is_misinform_apache[:, None], axis=0
    )
    blue_writes_misinform_tomcat = jnp.any(
        blue_target_mask & blue_is_misinform_tomcat[:, None], axis=0
    )
    blue_writes_misinform_sshd = jnp.any(
        blue_target_mask & blue_is_misinform_sshd[:, None], axis=0
    )
    blue_writes_acl = jnp.any(blue_target_mask & blue_is_acl[:, None], axis=0)
    new_decoy = jnp.where(blue_writes_decoy, jnp.int8(_DECOY_ACTIVE), state.hosts.decoy)
    new_decoy = jnp.where(
        blue_writes_misinform_apache, jnp.int8(_DECOY_APACHE), new_decoy
    )
    new_decoy = jnp.where(
        blue_writes_misinform_tomcat, jnp.int8(_DECOY_TOMCAT), new_decoy
    )
    new_decoy = jnp.where(blue_writes_misinform_sshd, jnp.int8(_DECOY_SSHD), new_decoy)
    new_edr = state.hosts.edr_active | blue_writes_acl
    new_honey = state.hosts.contains_honeytokens | blue_writes_honey

    already_compromised = state.hosts.system_integrity == jnp.int8(
        _INTEGRITY_COMPROMISED
    )
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
        blue_writes_restore_backup, jnp.int8(_INTEGRITY_CLEAN), new_integrity
    )

    new_human_vuln = jnp.where(
        blue_writes_sat,
        jnp.maximum(state.hosts.human_vulnerability - SAT_DROP, 0.0),
        state.hosts.human_vulnerability,
    )

    red_knowledge_writes = red_target_mask & red_is_recon[:, None]
    blue_knowledge_writes = (
        blue_target_mask & (blue_is_monitor | blue_is_analyze)[:, None]
    )
    new_red_knowledge = state.knowledge_mask[: spec.n_red] | red_knowledge_writes
    new_blue_knowledge = state.knowledge_mask[spec.n_red :] | blue_knowledge_writes

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
    blue_new_intel = (~state.knowledge_mask[spec.n_red :]) & blue_knowledge_writes
    red_new_intel = (~state.knowledge_mask[: spec.n_red]) & red_knowledge_writes
    n_cve_compromises = (
        red_is_bluekeep.sum() + red_is_eternalblue.sum() + red_is_http_rfi.sum()
    )

    exfil_targets = red_target_mask & red_is_exfil[:, None] & host_is_root[None, :]
    n_exfil_hosts = jnp.sum(jnp.any(exfil_targets, axis=0).astype(jnp.float32))

    was_decoy = state.hosts.decoy == jnp.int8(_DECOY_ACTIVE)
    new_decoy_deploys = blue_writes_decoy & ~was_decoy
    new_honey_deploys = blue_writes_honey & ~state.hosts.contains_honeytokens
    new_acl_writes = blue_writes_acl & ~state.hosts.edr_active
    new_misinform = (
        (blue_writes_misinform_apache & (state.hosts.decoy != jnp.int8(_DECOY_APACHE)))
        | (
            blue_writes_misinform_tomcat
            & (state.hosts.decoy != jnp.int8(_DECOY_TOMCAT))
        )
        | (blue_writes_misinform_sshd & (state.hosts.decoy != jnp.int8(_DECOY_SSHD)))
    )

    new_dc_compromise = (
        (new_compromised != state.hosts.compromised_by_id)
        & state.hosts.is_domain_controller
        & (new_compromised >= 0)
    )

    events = StepEvents(
        red_writes_user=red_writes_user,
        red_writes_root=red_writes_root,
        new_red_impacts=new_red_impacts,
        new_red_kinetics=new_red_kinetics,
        new_dc_compromise=new_dc_compromise,
        red_new_intel=red_new_intel,
        red_is_share=red_is_share,
        red_writes_kill=red_writes_kill,
        red_trapped=red_trapped,
        looted_per_agent=looted_per_agent,
        n_cve_compromises=n_cve_compromises,
        n_exfil_hosts=n_exfil_hosts,
        new_blue_isolations=new_blue_isolations,
        blue_writes_any_restore=blue_writes_any_restore,
        blue_writes_sat=blue_writes_sat,
        blue_is_rotate=blue_is_rotate,
        blue_new_intel=blue_new_intel,
        new_decoy_deploys=new_decoy_deploys,
        new_honey_deploys=new_honey_deploys,
        new_acl_writes=new_acl_writes,
        new_misinform=new_misinform,
    )
    reward = compute_rewards(state, events, spec)

    exfil_bytes_this_tick = EXFIL_PER_HOST * n_exfil_hosts
    new_state = replace(
        state,
        hosts=new_hosts,
        current_tick=state.current_tick + 1,
        knowledge_mask=new_knowledge,
        exfiltrated_bytes=state.exfiltrated_bytes + exfil_bytes_this_tick,
        agent_credentials=new_agent_credentials,
        in_flight_actions=new_queue,
    )
    return new_state, reward


def scenario_done(state, spec):
    """Per-env terminal flag for the active scenario. Vectorized over the batch."""
    hosts = state.hosts
    infected = hosts.privilege >= jnp.int8(_PRIV_USER)
    isolated = hosts.status == jnp.int8(_STATUS_ISOLATED)
    compromised = hosts.compromised_by_id >= 0
    kinetic = hosts.system_integrity == jnp.int8(_INTEGRITY_KINETIC)
    is_dc = hosts.is_domain_controller

    any_kinetic = jnp.any(kinetic, axis=-1)

    any_infected = jnp.any(infected, axis=-1)
    all_infected_isolated = jnp.all(~infected | isolated, axis=-1)
    apt_done = any_infected & all_infected_isolated

    any_dc = jnp.any(is_dc, axis=-1)
    all_dc_compromised = jnp.all(~is_dc | compromised, axis=-1)
    dc_done = any_dc & all_dc_compromised

    compromised_frac = jnp.mean(compromised.astype(jnp.float32), axis=-1)
    ransomware_done = (compromised_frac >= 0.9) | any_kinetic

    if spec.scenario == SCENARIO_APT:
        return apt_done
    if spec.scenario in (SCENARIO_CLOUD, SCENARIO_IOT):
        return dc_done
    if spec.scenario == SCENARIO_OT:
        return any_kinetic
    return ransomware_done
