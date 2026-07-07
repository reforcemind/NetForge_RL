from __future__ import annotations

from dataclasses import replace

import numpy as np

from netforge_rl.backends.jax.action_codes import (
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
from netforge_rl.backends.jax.scenario_config import (
    _BLUE_SCALE,
    _RED_SCALE,
    _RW,
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


def _resolve_conflicts_mask(red_mask, blue_mask, red_attempt, blue_attempt):
    """Red attempt fails if it shares a target with any active blue attempt."""
    blue_targets = np.any(blue_mask, axis=0)  # [N_HOSTS]
    red_hits_defended = np.any(red_mask & blue_targets[None, :], axis=1)  # [N_RED]
    return red_attempt & ~red_hits_defended


def reference_step(state, actions, spec):
    """Single-environment (unbatched) NumPy step. ``state`` is a JaxEnvState whose
    leaves are plain numpy arrays (no batch axis). Returns (new_state, rewards)."""
    n_hosts = spec.n_hosts
    hosts = state.hosts

    rt = np.asarray(actions.red_target_idx)
    bt = np.asarray(actions.blue_target_idx)
    ra = np.asarray(actions.red_attempt)
    ba = np.asarray(actions.blue_attempt)
    rat = np.asarray(actions.red_action_type)
    bat = np.asarray(actions.blue_action_type)

    def one_hot(idx, attempt):
        m = np.zeros((idx.shape[0], n_hosts), dtype=bool)
        m[np.arange(idx.shape[0]), idx] = True
        return m & attempt[:, None]

    red_mask = one_hot(rt, ra)
    blue_mask = one_hot(bt, ba)
    red_success = _resolve_conflicts_mask(red_mask, blue_mask, ra, ba)

    priv = np.asarray(hosts.privilege)
    status = np.asarray(hosts.status)
    integrity = np.asarray(hosts.system_integrity)
    compromised = np.asarray(hosts.compromised_by_id)
    decoy = np.asarray(hosts.decoy)
    vuln_mask = np.asarray(hosts.vuln_mask)
    host_tokens = np.asarray(hosts.host_tokens)
    os_family = np.asarray(hosts.os_family)
    creds = np.asarray(state.agent_credentials)

    def red_is(code):
        return (rat == code) & red_success

    red_is_compromise = red_is(RED_COMPROMISE)
    red_is_privesc = red_is(RED_PRIVESC)
    red_is_impact = red_is(RED_IMPACT)
    red_is_kinetic = red_is(RED_KINETIC)
    red_is_recon = (
        (rat == RED_RECON)
        | (rat == RED_NETWORK_SCAN)
        | (rat == RED_DISCOVER_REMOTE_SYSTEMS)
    ) & red_success
    red_is_exfil = red_is(RED_EXFILTRATE)
    red_is_share = red_is(RED_SHARE_INTEL)
    red_is_lsass = red_is(RED_DUMP_LSASS)
    red_is_pth = red_is(RED_PASS_THE_HASH)
    red_is_ptt = red_is(RED_PASS_THE_TICKET)
    red_is_juicy = red_is(RED_JUICY_POTATO)
    red_is_v4l2 = red_is(RED_V4L2)
    red_is_kill = red_is(RED_KILL_PROCESS)

    def cve_compromise(code, cve_idx):
        return red_is(code) & vuln_mask[rt, cve_idx]

    red_is_bluekeep = cve_compromise(RED_EXPLOIT_BLUEKEEP, _CVE_BLUEKEEP)
    red_is_eternalblue = cve_compromise(RED_EXPLOIT_ETERNALBLUE, _CVE_ETERNALBLUE)
    red_is_http_rfi = cve_compromise(RED_EXPLOIT_HTTP_RFI, _CVE_HTTP_RFI)
    red_is_ip_frag = red_is(RED_IP_FRAGMENTATION)
    spearphish_ok = red_is(RED_SPEARPHISHING) & (os_family[rt] == OS_WINDOWS)
    red_is_compromise = (
        red_is_compromise
        | red_is_bluekeep
        | red_is_eternalblue
        | red_is_http_rfi
        | red_is_ip_frag
        | spearphish_ok
    )

    def blue_is(code):
        return (bat == code) & ba

    blue_is_isolate = blue_is(BLUE_ISOLATE)
    blue_is_restore = blue_is(BLUE_RESTORE)
    blue_is_restore_backup = blue_is(BLUE_RESTORE_FROM_BACKUP)
    blue_is_decoy = blue_is(BLUE_DEPLOY_DECOY)
    blue_is_honey = blue_is(BLUE_DEPLOY_HONEYTOKEN)
    blue_is_remove = blue_is(BLUE_REMOVE)
    blue_is_sat = blue_is(BLUE_SAT)
    blue_is_monitor = blue_is(BLUE_MONITOR)
    blue_is_analyze = blue_is(BLUE_ANALYZE)
    blue_is_acl = blue_is(BLUE_CONFIGURE_ACL)
    blue_is_rotate = blue_is(BLUE_ROTATE_KERBEROS)
    blue_is_mis_apache = blue_is(BLUE_MISINFORM_APACHE)
    blue_is_mis_tomcat = blue_is(BLUE_MISINFORM_TOMCAT)
    blue_is_mis_sshd = blue_is(BLUE_MISINFORM_SSHD)

    def writes(mask_2d, flags):
        return np.any(mask_2d & flags[:, None], axis=0)

    blue_writes_isolate = writes(blue_mask, blue_is_isolate)
    blue_writes_restore = writes(blue_mask, blue_is_restore)
    blue_writes_restore_backup = writes(blue_mask, blue_is_restore_backup)
    blue_writes_any_restore = blue_writes_restore | blue_writes_restore_backup
    blue_writes_decoy = writes(blue_mask, blue_is_decoy)
    blue_writes_honey = writes(blue_mask, blue_is_honey)
    blue_writes_remove = writes(blue_mask, blue_is_remove)
    blue_writes_sat = writes(blue_mask, blue_is_sat)
    blue_writes_acl = writes(blue_mask, blue_is_acl)
    blue_writes_mis_apache = writes(blue_mask, blue_is_mis_apache)
    blue_writes_mis_tomcat = writes(blue_mask, blue_is_mis_tomcat)
    blue_writes_mis_sshd = writes(blue_mask, blue_is_mis_sshd)

    red_writes_user = writes(red_mask, red_is_compromise)
    red_writes_root = writes(red_mask, red_is_privesc)
    red_writes_impact = writes(red_mask, red_is_impact)
    red_writes_kinetic = writes(red_mask, red_is_kinetic)

    host_already_owned = priv >= _PRIV_USER
    host_is_root = priv == _PRIV_ROOT
    red_writes_root = red_writes_root & host_already_owned
    red_writes_impact = red_writes_impact & host_is_root
    red_writes_kinetic = red_writes_kinetic & host_is_root

    n_red = spec.n_red
    lsass_attempts = red_is_lsass & host_is_root[rt]
    looted = np.where(
        lsass_attempts[:, None], host_tokens[rt], np.zeros_like(creds[:n_red])
    )
    new_red_creds = creds[:n_red] | looted

    red_creds = creds[:n_red]
    target_tokens = host_tokens[rt]
    held = np.where(target_tokens, red_creds, True).all(axis=-1)
    target_has_req = target_tokens.any(axis=-1)
    red_token_ok = held & target_has_req

    red_writes_user |= writes(red_mask, red_is_pth & red_token_ok)
    red_writes_root |= writes(red_mask, red_is_ptt & red_token_ok)

    juicy_ok = red_is_juicy & (os_family[rt] == OS_WINDOWS)
    v4l2_ok = red_is_v4l2 & (os_family[rt] == OS_LINUX)
    red_writes_root = (
        red_writes_root | writes(red_mask, juicy_ok) | writes(red_mask, v4l2_ok)
    )

    red_writes_kill = writes(red_mask, red_is_kill) & host_is_root

    any_blue_rotate = np.any(blue_is_rotate)
    if any_blue_rotate:
        new_red_creds = np.zeros_like(new_red_creds)
    new_agent_credentials = np.concatenate([new_red_creds, creds[n_red:]], axis=0)

    already_isolated = status == _STATUS_ISOLATED
    new_blue_isolations = blue_writes_isolate & ~already_isolated

    new_status = status.copy()
    new_status = np.where(blue_writes_any_restore, _STATUS_ONLINE, new_status)
    new_status = np.where(blue_writes_isolate, _STATUS_ISOLATED, new_status)
    new_status = np.where(red_writes_kill, _STATUS_KERNEL_PANIC, new_status)

    new_priv = priv.copy()
    new_priv = np.where(red_writes_user, _PRIV_USER, new_priv)
    new_priv = np.where(red_writes_root, _PRIV_ROOT, new_priv)
    new_priv = np.where(blue_writes_any_restore, _PRIV_NONE, new_priv)
    new_priv = np.where(blue_writes_remove, _PRIV_NONE, new_priv)

    red_owners = red_mask & (red_is_compromise | (red_is_pth & red_token_ok))[:, None]
    any_red_owns = np.any(red_owners, axis=0)
    chosen_red = np.argmax(red_owners.astype(np.int8), axis=0).astype(np.int8)
    new_compromised = np.where(any_red_owns, chosen_red, compromised)
    new_compromised = np.where(blue_writes_any_restore, np.int8(-1), new_compromised)

    new_decoy = np.where(blue_writes_decoy, _DECOY_ACTIVE, decoy)
    new_decoy = np.where(blue_writes_mis_apache, _DECOY_APACHE, new_decoy)
    new_decoy = np.where(blue_writes_mis_tomcat, _DECOY_TOMCAT, new_decoy)
    new_decoy = np.where(blue_writes_mis_sshd, _DECOY_SSHD, new_decoy)
    new_edr = np.asarray(hosts.edr_active) | blue_writes_acl
    new_honey = np.asarray(hosts.contains_honeytokens) | blue_writes_honey

    already_compromised = integrity == _INTEGRITY_COMPROMISED
    already_kinetic = integrity == _INTEGRITY_KINETIC
    new_red_impacts = red_writes_impact & ~already_compromised & ~already_kinetic
    new_red_kinetics = red_writes_kinetic & ~already_kinetic

    new_integrity = integrity.copy()
    new_integrity = np.where(red_writes_impact, _INTEGRITY_COMPROMISED, new_integrity)
    new_integrity = np.where(red_writes_kinetic, _INTEGRITY_KINETIC, new_integrity)
    new_integrity = np.where(
        blue_writes_restore_backup, _INTEGRITY_CLEAN, new_integrity
    )

    new_human_vuln = np.where(
        blue_writes_sat,
        np.maximum(np.asarray(hosts.human_vulnerability) - SAT_DROP, 0.0),
        np.asarray(hosts.human_vulnerability),
    )

    knowledge = np.asarray(state.knowledge_mask)
    red_knowledge_writes = red_mask & red_is_recon[:, None]
    blue_knowledge_writes = blue_mask & (blue_is_monitor | blue_is_analyze)[:, None]
    new_red_knowledge = knowledge[:n_red] | red_knowledge_writes
    new_blue_knowledge = knowledge[n_red:] | blue_knowledge_writes

    if np.any(red_is_share):
        red_union = np.any(new_red_knowledge, axis=0)
        new_red_knowledge = np.broadcast_to(
            red_union[None, :], new_red_knowledge.shape
        ).copy()
    new_knowledge = np.concatenate([new_red_knowledge, new_blue_knowledge], axis=0)

    red_trapped = np.any(
        red_mask
        & red_is_compromise[:, None]
        & np.asarray(hosts.contains_honeytokens)[None, :],
        axis=1,
    )
    blue_new_intel = (~knowledge[n_red:]) & blue_knowledge_writes
    red_new_intel = (~knowledge[:n_red]) & red_knowledge_writes
    n_cve_compromises = (
        red_is_bluekeep.sum() + red_is_eternalblue.sum() + red_is_http_rfi.sum()
    )

    exfil_targets = red_mask & red_is_exfil[:, None] & host_is_root[None, :]
    n_exfil_hosts = float(np.sum(np.any(exfil_targets, axis=0).astype(np.float32)))

    was_decoy = decoy == _DECOY_ACTIVE
    new_decoy_deploys = blue_writes_decoy & ~was_decoy
    new_honey_deploys = blue_writes_honey & ~np.asarray(hosts.contains_honeytokens)
    new_acl_writes = blue_writes_acl & ~np.asarray(hosts.edr_active)
    new_misinform = (
        (blue_writes_mis_apache & (decoy != _DECOY_APACHE))
        | (blue_writes_mis_tomcat & (decoy != _DECOY_TOMCAT))
        | (blue_writes_mis_sshd & (decoy != _DECOY_SSHD))
    )

    is_dc = np.asarray(hosts.is_domain_controller)
    new_dc_compromise = (
        (new_compromised != compromised) & is_dc & (new_compromised >= 0)
    )

    rw_red, rw_blue = _RW[spec.scenario]
    w_user, w_root, w_impact, w_kinetic, w_exfil, w_dc, w_recon = rw_red
    w_good, w_bad, w_restore, w_health, w_dcloss, w_deceive = rw_blue

    target_clean = compromised < 0
    bad_isolations = new_blue_isolations & target_clean
    good_isolations = new_blue_isolations & ~target_clean

    healthy_ratio = float(
        np.sum(((compromised < 0) & (status == _STATUS_ONLINE)).astype(np.float32))
        / float(n_hosts)
    )
    dc_lost = float(np.any(is_dc & (compromised >= 0)))

    raw_blue = (
        w_good * good_isolations.sum()
        - w_bad * bad_isolations.sum()
        + w_restore * blue_writes_any_restore.sum()
        + w_health * healthy_ratio
        - w_dcloss * dc_lost
        + w_deceive
        * (new_decoy_deploys | new_honey_deploys | new_acl_writes | new_misinform).sum()
        + 0.3 * blue_writes_sat.sum()
        + 0.2 * blue_new_intel.sum()
        + 4.0 * blue_is_rotate.sum()
    )
    blue_reward = float(np.tanh(raw_blue / _BLUE_SCALE))

    raw_red = (
        w_user * red_writes_user.sum()
        + 0.5 * float(n_cve_compromises)
        + w_root * red_writes_root.sum()
        + w_impact * new_red_impacts.sum()
        + w_kinetic * new_red_kinetics.sum()
        + w_dc * new_dc_compromise.sum()
        + w_recon * red_new_intel.sum()
        + w_exfil * n_exfil_hosts
        + 0.4 * red_is_share.sum()
        + 2.0 * np.any(looted, axis=-1).sum()
        + 5.0 * red_writes_kill.sum()
    )
    red_team_reward = float(np.tanh(raw_red / _RED_SCALE))
    red_trap_penalty = -0.5 * red_trapped.astype(np.float32)

    exfil_bytes = EXFIL_PER_HOST * n_exfil_hosts

    new_hosts = replace(
        hosts,
        status=new_status,
        privilege=new_priv,
        compromised_by_id=new_compromised,
        decoy=new_decoy,
        contains_honeytokens=new_honey,
        system_integrity=new_integrity,
        human_vulnerability=new_human_vuln,
        edr_active=new_edr,
    )
    new_state = replace(
        state,
        hosts=new_hosts,
        current_tick=state.current_tick + 1,
        knowledge_mask=new_knowledge,
        exfiltrated_bytes=state.exfiltrated_bytes + exfil_bytes,
        agent_credentials=new_agent_credentials,
    )

    red_rewards = (
        np.full((n_red,), red_team_reward, dtype=np.float32) + red_trap_penalty
    )
    blue_rewards = np.full((spec.n_blue,), blue_reward, dtype=np.float32)
    return new_state, np.concatenate([red_rewards, blue_rewards])


def reference_done(state, spec) -> bool:
    """NumPy mirror of ``scenario_done`` for a single (unbatched) state."""
    hosts = state.hosts
    priv = np.asarray(hosts.privilege)
    status = np.asarray(hosts.status)
    integrity = np.asarray(hosts.system_integrity)
    compromised = np.asarray(hosts.compromised_by_id)
    is_dc = np.asarray(hosts.is_domain_controller)

    infected = priv >= _PRIV_USER
    isolated = status == _STATUS_ISOLATED
    comp = compromised >= 0
    kinetic = integrity == _INTEGRITY_KINETIC

    any_kinetic = bool(np.any(kinetic))
    if spec.scenario == SCENARIO_APT:
        return bool(np.any(infected) and np.all(~infected | isolated))
    if spec.scenario in (SCENARIO_CLOUD, SCENARIO_IOT):
        return bool(np.any(is_dc) and np.all(~is_dc | comp))
    if spec.scenario == SCENARIO_OT:
        return any_kinetic
    return bool(np.mean(comp.astype(np.float32)) >= 0.9 or any_kinetic)
