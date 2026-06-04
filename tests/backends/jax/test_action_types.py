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
    BLUE_CONFIGURE_ACL,
    BLUE_DEPLOY_DECOY,
    BLUE_DEPLOY_HONEYTOKEN,
    BLUE_ISOLATE,
    BLUE_MISINFORM,
    BLUE_MONITOR,
    BLUE_REMOVE,
    BLUE_RESTORE,
    BLUE_ROTATE_KERBEROS,
    BLUE_SAT,
    EXFIL_PER_HOST,
    RED_COMPROMISE,
    RED_DUMP_LSASS,
    RED_EXFILTRATE,
    RED_EXPLOIT_BLUEKEEP,
    RED_EXPLOIT_ETERNALBLUE,
    RED_EXPLOIT_HTTP_RFI,
    RED_IMPACT,
    RED_JUICY_POTATO,
    RED_KILL_PROCESS,
    RED_KINETIC,
    RED_PASS_THE_HASH,
    RED_PASS_THE_TICKET,
    RED_PRIVESC,
    RED_V4L2,
    RED_RECON,
    RED_SHARE_INTEL,
    SAT_DROP,
)
from netforge_rl.core.functional import CVE_CODES
from netforge_rl.core.functional import (
    DECOY_CODES,
    INTEGRITY_CODES,
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


# ── Blue honeytoken + Red trap penalty ───────────────────────────────────


@pytest.mark.fast
def test_honeytoken_traps_red_compromise(global_state) -> None:
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    # 1. Blue deploys honeytoken on host 12.
    state, _ = step(state, _act(
        red_t=[[99]], blue_t=[[12]],
        red_a=[[False]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_DEPLOY_HONEYTOKEN]],
    ))
    assert bool(state.hosts.contains_honeytokens[0, 12])

    # 2. Red tries to compromise host 12 -> -5 trap penalty.
    _, rewards = step(state, _act(
        red_t=[[12]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_ISOLATE]],
    ))
    # Red 0's reward = +1 (compromise) - 5 (trap) = -4.
    assert float(rewards[0, 0]) == pytest.approx(-4.0)


# ── Red impact + kinetic ────────────────────────────────────────────────


def _own_to_root(state, step, idx: int):
    """Helper: drive a host from None -> User -> Root via two compromise+privesc steps."""
    state, _ = step(state, _act(
        red_t=[[idx]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_ISOLATE]],
    ))
    state, _ = step(state, _act(
        red_t=[[idx]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_PRIVESC]], blue_type=[[BLUE_ISOLATE]],
    ))
    return state


@pytest.mark.fast
def test_impact_requires_root(global_state) -> None:
    """Impact against an unowned host is a no-op."""
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)
    state, _ = step(state, _act(
        red_t=[[4]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_IMPACT]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.system_integrity[0, 4]) == INTEGRITY_CODES.index('clean')


@pytest.mark.fast
def test_impact_promotes_integrity_when_root(global_state) -> None:
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)
    state = _own_to_root(state, step, idx=8)
    state, rewards = step(state, _act(
        red_t=[[8]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_IMPACT]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.system_integrity[0, 8]) == INTEGRITY_CODES.index('compromised')
    assert float(rewards[0, 0]) == pytest.approx(10.0)


@pytest.mark.fast
def test_kinetic_super_reward(global_state) -> None:
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)
    state = _own_to_root(state, step, idx=9)
    state, rewards = step(state, _act(
        red_t=[[9]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_KINETIC]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.system_integrity[0, 9]) == INTEGRITY_CODES.index(
        'kinetic_destruction'
    )
    assert float(rewards[0, 0]) == pytest.approx(10_000.0)


@pytest.mark.fast
def test_restore_clears_integrity(global_state) -> None:
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)
    state = _own_to_root(state, step, idx=10)
    state, _ = step(state, _act(
        red_t=[[10]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_IMPACT]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.system_integrity[0, 10]) == INTEGRITY_CODES.index('compromised')
    state, _ = step(state, _act(
        red_t=[[99]], blue_t=[[10]],
        red_a=[[False]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_RESTORE]],
    ))
    assert int(state.hosts.system_integrity[0, 10]) == INTEGRITY_CODES.index('clean')


# ── Blue extras: REMOVE + SAT ───────────────────────────────────────────


@pytest.mark.fast
def test_remove_clears_priv_but_keeps_status(global_state) -> None:
    """REMOVE wipes privilege; unlike RESTORE it does NOT flip status."""
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    # Red compromises host 11.
    state, _ = step(state, _act(
        red_t=[[11]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, 11]) == PRIVILEGE_CODES.index('User')

    # Isolate host 11 so we can verify REMOVE keeps it isolated.
    state, _ = step(state, _act(
        red_t=[[99]], blue_t=[[11]],
        red_a=[[False]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.status[0, 11]) == STATUS_CODES.index('isolated')

    state, rewards = step(state, _act(
        red_t=[[99]], blue_t=[[11]],
        red_a=[[False]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_REMOVE]],
    ))
    assert int(state.hosts.privilege[0, 11]) == PRIVILEGE_CODES.index('None')
    assert int(state.hosts.status[0, 11]) == STATUS_CODES.index('isolated')  # unchanged
    # Blue reward 0 gets the +1.5 remove bonus.
    assert float(rewards[0, spec.n_red]) == pytest.approx(1.5)


@pytest.mark.fast
def test_sat_decrements_human_vulnerability(global_state) -> None:
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    # Pick a host with high human_vulnerability to make the assertion stable.
    idx = int(state.hosts.human_vulnerability[0].argmax())
    before = float(state.hosts.human_vulnerability[0, idx])

    state, rewards = step(state, _act(
        red_t=[[99]], blue_t=[[idx]],
        red_a=[[False]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_SAT]],
    ))
    after = float(state.hosts.human_vulnerability[0, idx])
    assert after == pytest.approx(max(before - SAT_DROP, 0.0))
    assert float(rewards[0, spec.n_red]) == pytest.approx(0.3)


# ── CVE-gated exploits ──────────────────────────────────────────────────


def _find_vulnerable_host(state, cve_name: str) -> int | None:
    """Return a host index whose vuln_mask has the named CVE bit set, or None."""
    col = CVE_CODES.index(cve_name)
    mask = state.hosts.vuln_mask[0, :, col]  # batch=0
    candidates = [int(i) for i, v in enumerate(mask) if bool(v)]
    return candidates[0] if candidates else None


@pytest.mark.fast
def test_bluekeep_needs_cve_bit_set(global_state) -> None:
    """RED_EXPLOIT_BLUEKEEP against a host without CVE-2019-0708 is a no-op."""
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    # Find a host that does NOT have BlueKeep.
    col = CVE_CODES.index('CVE-2019-0708')
    no_bluekeep = [
        int(i) for i in range(100)
        if not bool(state.hosts.vuln_mask[0, i, col])
        and int(state.hosts.privilege[0, i]) == 0
    ]
    if not no_bluekeep:
        pytest.skip('topology had no BlueKeep-free host')
    idx = no_bluekeep[0]
    before = int(state.hosts.privilege[0, idx])
    state, _ = step(state, _act(
        red_t=[[idx]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_EXPLOIT_BLUEKEEP]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, idx]) == before


@pytest.mark.fast
def test_cve_exploit_compromises_when_bit_set(global_state) -> None:
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    # Pick any host and force a known CVE bit on for it (mutate the
    # batched state — safe inside a test).
    import jax.numpy as jnp
    col = CVE_CODES.index('MS17-010')
    new_vm = state.hosts.vuln_mask.at[0, 13, col].set(True)
    state = state.__class__(
        **{**state.__dict__, 'hosts': state.hosts.__class__(
            **{**state.hosts.__dict__, 'vuln_mask': new_vm}
        )}
    )
    state, rewards = step(state, _act(
        red_t=[[13]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_EXPLOIT_ETERNALBLUE]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, 13]) == PRIVILEGE_CODES.index('User')
    # Compromise (+1) + CVE bonus (+0.5) = 1.5.
    assert float(rewards[0, 0]) == pytest.approx(1.5)


# ── Knowledge mask + recon ──────────────────────────────────────────────


@pytest.mark.fast
def test_recon_sets_red_knowledge_bit(global_state) -> None:
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    assert not bool(state.knowledge_mask[0, 0, 23])  # red row 0, host 23
    state, rewards = step(state, _act(
        red_t=[[23]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_RECON]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert bool(state.knowledge_mask[0, 0, 23])
    # +0.2 intel reward on first sighting.
    assert float(rewards[0, 0]) == pytest.approx(0.2)


@pytest.mark.fast
def test_monitor_sets_blue_knowledge_bit(global_state) -> None:
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    # Blue row 0 starts at agent_ids index N_RED=1.
    assert not bool(state.knowledge_mask[0, 1, 42])
    state, _ = step(state, _act(
        red_t=[[99]], blue_t=[[42]],
        red_a=[[False]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_MONITOR]],
    ))
    assert bool(state.knowledge_mask[0, 1, 42])


@pytest.mark.fast
def test_intel_reward_only_on_first_sighting(global_state) -> None:
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    state, r_first = step(state, _act(
        red_t=[[5]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_RECON]], blue_type=[[BLUE_ISOLATE]],
    ))
    state, r_second = step(state, _act(
        red_t=[[5]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_RECON]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert float(r_first[0, 0]) > 0.0
    # Already known -> no intel reward on repeat.
    assert float(r_second[0, 0]) == pytest.approx(0.0)


@pytest.mark.fast
def test_misinform_brands_decoy_as_apache(global_state):
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)
    idx = int(
        next(
            i for i in range(100)
            if int(state.hosts.decoy[0, i]) == DECOY_CODES.index('inactive')
        )
    )
    state, _ = step(state, _act(
        red_t=[[99]], blue_t=[[idx]],
        red_a=[[False]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_MISINFORM]],
    ))
    assert int(state.hosts.decoy[0, idx]) == DECOY_CODES.index('Apache')


@pytest.mark.fast
def test_exfiltrate_requires_root(global_state):
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)
    before = float(state.exfiltrated_bytes[0])
    state, rewards = step(state, _act(
        red_t=[[6]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_EXFILTRATE]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert float(state.exfiltrated_bytes[0]) == before  # not Root -> no bytes
    assert float(rewards[0, 0]) == pytest.approx(0.0)


@pytest.mark.fast
def test_exfiltrate_accumulates_when_root(global_state):
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)
    state = _own_to_root(state, step, idx=14)
    before = float(state.exfiltrated_bytes[0])
    state, rewards = step(state, _act(
        red_t=[[14]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_EXFILTRATE]], blue_type=[[BLUE_ISOLATE]],
    ))
    after = float(state.exfiltrated_bytes[0])
    assert after == pytest.approx(before + EXFIL_PER_HOST)
    assert float(rewards[0, 0]) == pytest.approx(EXFIL_PER_HOST)


@pytest.mark.fast
def test_configure_acl_flips_edr_active(global_state):
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)
    assert not bool(state.hosts.edr_active[0, 22])
    state, rewards = step(state, _act(
        red_t=[[99]], blue_t=[[22]],
        red_a=[[False]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_CONFIGURE_ACL]],
    ))
    assert bool(state.hosts.edr_active[0, 22])
    assert float(rewards[0, spec.n_red]) == pytest.approx(0.7)


@pytest.mark.fast
def test_share_intel_broadcasts_red_knowledge(global_state):
    spec = _spec(n_red=2, n_blue=1)
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    # Red 0 recons host 3; Red 1 does nothing.
    state, _ = step(state, _act(
        red_t=[[3, 0]], blue_t=[[99]],
        red_a=[[True, False]], blue_a=[[False]],
        red_type=[[RED_RECON, RED_COMPROMISE]],
        blue_type=[[BLUE_ISOLATE]],
    ))
    assert bool(state.knowledge_mask[0, 0, 3])
    assert not bool(state.knowledge_mask[0, 1, 3])

    # Red 1 shares -> Red 0's bit propagates to Red 1.
    state, _ = step(state, _act(
        red_t=[[0, 0]], blue_t=[[99]],
        red_a=[[False, True]], blue_a=[[False]],
        red_type=[[RED_COMPROMISE, RED_SHARE_INTEL]],
        blue_type=[[BLUE_ISOLATE]],
    ))
    assert bool(state.knowledge_mask[0, 1, 3])


@pytest.mark.fast
def test_lsass_loots_token_when_root(global_state):
    """DumpLSASS on a Rooted host that carries an admin token loots it."""
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    # Find a host that has at least one token in host_tokens.
    candidates = [i for i in range(100) if bool(state.hosts.host_tokens[0, i].any())]
    if not candidates:
        pytest.skip('no host with tokens in this seed')
    idx = candidates[0]

    state = _own_to_root(state, step, idx=idx)
    before_creds = bool(state.agent_credentials[0, 0].any())
    state, _ = step(state, _act(
        red_t=[[idx]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_DUMP_LSASS]], blue_type=[[BLUE_ISOLATE]],
    ))
    after_creds = bool(state.agent_credentials[0, 0].any())
    assert after_creds and not before_creds


@pytest.mark.fast
def test_pass_the_hash_compromises_with_token(global_state):
    """Audit fix 1.1: PTH requires the target's required tokens, not just any."""
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    src = next(
        (i for i in range(100) if bool(state.hosts.host_tokens[0, i].any())),
        None,
    )
    if src is None:
        pytest.skip('no host with tokens in this seed')

    state = _own_to_root(state, step, idx=src)
    state, _ = step(state, _act(
        red_t=[[src]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_DUMP_LSASS]], blue_type=[[BLUE_ISOLATE]],
    ))
    # Token locality: PTH only unlocks hosts whose required tokens are a
    # subset of what Red looted.
    src_tokens = state.hosts.host_tokens[0, src]
    tgt = next(
        (
            i for i in range(100)
            if i != src
            and int(state.hosts.privilege[0, i]) == PRIVILEGE_CODES.index('None')
            and bool(state.hosts.host_tokens[0, i].any())
            and bool((state.hosts.host_tokens[0, i] & ~src_tokens).any()) is False
        ),
        None,
    )
    if tgt is None:
        pytest.skip('no PTH-reachable target on this seed')
    state, _ = step(state, _act(
        red_t=[[tgt]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_PASS_THE_HASH]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, tgt]) == PRIVILEGE_CODES.index('User')


@pytest.mark.fast
def test_pass_the_hash_rejected_on_wrong_token(global_state):
    """Audit fix 1.1: PTH on a host requiring a different token must no-op."""
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    # Find a (src, tgt) pair where the target requires a token the source doesn't yield.
    pair = None
    for src in range(100):
        if not bool(state.hosts.host_tokens[0, src].any()):
            continue
        src_tokens = state.hosts.host_tokens[0, src]
        for tgt in range(100):
            if tgt == src:
                continue
            tgt_tokens = state.hosts.host_tokens[0, tgt]
            if not bool(tgt_tokens.any()):
                continue
            if bool((tgt_tokens & ~src_tokens).any()):
                pair = (src, tgt)
                break
        if pair:
            break
    if pair is None:
        pytest.skip('no token-mismatch pair on this seed')

    src, tgt = pair
    state = _own_to_root(state, step, idx=src)
    state, _ = step(state, _act(
        red_t=[[src]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_DUMP_LSASS]], blue_type=[[BLUE_ISOLATE]],
    ))
    before = int(state.hosts.privilege[0, tgt])
    state, _ = step(state, _act(
        red_t=[[tgt]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_PASS_THE_HASH]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, tgt]) == before


@pytest.mark.fast
def test_pass_the_hash_no_op_without_token(global_state):
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)
    clean_idx = next(
        i for i in range(100)
        if int(state.hosts.privilege[0, i]) == PRIVILEGE_CODES.index('None')
    )
    state, _ = step(state, _act(
        red_t=[[clean_idx]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_PASS_THE_HASH]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, clean_idx]) == PRIVILEGE_CODES.index('None')


@pytest.mark.fast
def test_rotate_kerberos_clears_red_credentials(global_state):
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)
    candidates = [i for i in range(100) if bool(state.hosts.host_tokens[0, i].any())]
    if not candidates:
        pytest.skip('no host with tokens in this seed')
    state = _own_to_root(state, step, idx=candidates[0])
    state, _ = step(state, _act(
        red_t=[[candidates[0]]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_DUMP_LSASS]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert bool(state.agent_credentials[0, 0].any())

    state, _ = step(state, _act(
        red_t=[[99]], blue_t=[[0]],
        red_a=[[False]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_ROTATE_KERBEROS]],
    ))
    assert not bool(state.agent_credentials[0, 0].any())


@pytest.mark.fast
def test_sat_clamps_at_zero(global_state) -> None:
    """Repeated SAT shouldn't drive vulnerability negative."""
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    idx = int(state.hosts.human_vulnerability[0].argmax())
    for _ in range(50):
        state, _ = step(state, _act(
            red_t=[[99]], blue_t=[[idx]],
            red_a=[[False]], blue_a=[[True]],
            red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_SAT]],
        ))
    assert float(state.hosts.human_vulnerability[0, idx]) == 0.0


# ── OS-gated privesc + KillProcess + audit reward gates ─────────────────


def _find_host_by_os(state, family_code):
    for i in range(100):
        if int(state.hosts.os_family[0, i]) == family_code:
            return i
    return None


@pytest.mark.fast
def test_juicy_potato_needs_windows(global_state):
    from netforge_rl.core.functional import OS_LINUX, OS_WINDOWS

    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    win_idx = _find_host_by_os(state, OS_WINDOWS)
    linux_idx = _find_host_by_os(state, OS_LINUX)
    if win_idx is None or linux_idx is None:
        pytest.skip('no Windows or Linux host on this seed')

    # First compromise both to User so privesc is meaningful.
    for idx in (win_idx, linux_idx):
        state, _ = step(state, _act(
            red_t=[[idx]], blue_t=[[99]],
            red_a=[[True]], blue_a=[[False]],
            red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_ISOLATE]],
        ))

    # JuicyPotato on Windows succeeds, on Linux it doesn't.
    state, _ = step(state, _act(
        red_t=[[win_idx]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_JUICY_POTATO]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, win_idx]) == PRIVILEGE_CODES.index('Root')

    state, _ = step(state, _act(
        red_t=[[linux_idx]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_JUICY_POTATO]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, linux_idx]) == PRIVILEGE_CODES.index('User')


@pytest.mark.fast
def test_v4l2_needs_linux(global_state):
    from netforge_rl.core.functional import OS_LINUX, OS_WINDOWS

    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    linux_idx = _find_host_by_os(state, OS_LINUX)
    win_idx = _find_host_by_os(state, OS_WINDOWS)
    if linux_idx is None or win_idx is None:
        pytest.skip('no Linux or Windows host on this seed')

    for idx in (linux_idx, win_idx):
        state, _ = step(state, _act(
            red_t=[[idx]], blue_t=[[99]],
            red_a=[[True]], blue_a=[[False]],
            red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_ISOLATE]],
        ))

    state, _ = step(state, _act(
        red_t=[[linux_idx]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_V4L2]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, linux_idx]) == PRIVILEGE_CODES.index('Root')

    state, _ = step(state, _act(
        red_t=[[win_idx]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_V4L2]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.privilege[0, win_idx]) == PRIVILEGE_CODES.index('User')


@pytest.mark.fast
def test_kill_process_panics_root_host(global_state):
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)
    state = _own_to_root(state, step, idx=25)

    state, rewards = step(state, _act(
        red_t=[[25]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_KILL_PROCESS]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert int(state.hosts.status[0, 25]) == STATUS_CODES.index('kernel_panic')
    # KillProcess pays 5.0 + the Red team's intel base; just check >=5.
    assert float(rewards[0, 0]) >= 5.0


@pytest.mark.fast
def test_audit_no_reward_for_reisolating_already_isolated(global_state):
    """Audit fix 1.2: re-isolating an isolated host should not pay Blue again."""
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)

    state, _ = step(state, _act(
        red_t=[[99]], blue_t=[[18]],
        red_a=[[False]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_ISOLATE]],
    ))
    state, rewards = step(state, _act(
        red_t=[[99]], blue_t=[[18]],
        red_a=[[False]], blue_a=[[True]],
        red_type=[[RED_COMPROMISE]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert float(rewards[0, spec.n_red]) == pytest.approx(0.0)


@pytest.mark.fast
def test_audit_no_reward_for_reimpacting_compromised_host(global_state):
    """Audit fix 1.2: re-impacting a compromised host should not pay Red again."""
    spec = _spec()
    state = _state(global_state, batch=1)
    step = make_vector_step(spec)
    state = _own_to_root(state, step, idx=33)

    state, r_first = step(state, _act(
        red_t=[[33]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_IMPACT]], blue_type=[[BLUE_ISOLATE]],
    ))
    state, r_second = step(state, _act(
        red_t=[[33]], blue_t=[[99]],
        red_a=[[True]], blue_a=[[False]],
        red_type=[[RED_IMPACT]], blue_type=[[BLUE_ISOLATE]],
    ))
    assert float(r_first[0, 0]) > 0.0
    assert float(r_second[0, 0]) == pytest.approx(0.0)
