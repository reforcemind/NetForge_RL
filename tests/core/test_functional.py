"""Round-trip and invariant tests for the frozen EnvState scaffolding."""

import dataclasses

import numpy as np
import pytest

from netforge_rl.core.functional import (
    DECOY_CODES,
    EnvState,
    HostArrays,
    N_HOSTS,
    PRIVILEGE_CODES,
    STATUS_CODES,
    from_global_state,
    to_global_state,
)


AGENTS = (
    'red_operator',
    'blue_dmz',
    'blue_internal',
    'blue_restricted',
)


@pytest.mark.fast
def test_from_global_state_produces_expected_shapes(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)

    assert isinstance(snap, EnvState)
    assert snap.host_count == N_HOSTS
    assert snap.agent_ids == AGENTS

    for arr in (
        snap.hosts.status,
        snap.hosts.privilege,
        snap.hosts.decoy,
        snap.hosts.compromised_by_id,
    ):
        assert arr.shape == (N_HOSTS,)
        assert arr.dtype == np.int8

    for arr in (
        snap.hosts.edr_active,
        snap.hosts.is_domain_controller,
        snap.hosts.contains_honeytokens,
    ):
        assert arr.shape == (N_HOSTS,)
        assert arr.dtype == bool

    for arr in (snap.hosts.human_vulnerability, snap.hosts.cvss_score):
        assert arr.shape == (N_HOSTS,)
        assert arr.dtype == np.float32

    for arr in (
        snap.agent_energy,
        snap.agent_funds,
        snap.agent_compute,
        snap.agent_locked_until,
    ):
        assert arr.shape == (len(AGENTS),)
        assert arr.dtype == np.int32


@pytest.mark.fast
def test_envstate_is_frozen(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.current_tick = 999  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        from netforge_rl.core.functional import N_CVE, N_TOKEN

        snap.hosts = HostArrays(  # type: ignore[misc]
            status=np.zeros(N_HOSTS, dtype=np.int8),
            privilege=np.zeros(N_HOSTS, dtype=np.int8),
            decoy=np.zeros(N_HOSTS, dtype=np.int8),
            edr_active=np.zeros(N_HOSTS, dtype=bool),
            is_domain_controller=np.zeros(N_HOSTS, dtype=bool),
            contains_honeytokens=np.zeros(N_HOSTS, dtype=bool),
            human_vulnerability=np.zeros(N_HOSTS, dtype=np.float32),
            cvss_score=np.zeros(N_HOSTS, dtype=np.float32),
            compromised_by_id=np.zeros(N_HOSTS, dtype=np.int8),
            system_integrity=np.zeros(N_HOSTS, dtype=np.int8),
            vuln_mask=np.zeros((N_HOSTS, N_CVE), dtype=bool),
            host_tokens=np.zeros((N_HOSTS, N_TOKEN), dtype=bool),
            os_family=np.zeros(N_HOSTS, dtype=np.int8),
        )


@pytest.mark.fast
def test_host_ordering_matches_legacy_sorted_ips(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    assert list(snap.meta.ip) == sorted(global_state.all_hosts.keys())


@pytest.mark.fast
def test_with_tick_returns_new_instance(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    bumped = snap.with_tick(42)
    assert snap.current_tick == 0  # original untouched
    assert bumped.current_tick == 42
    assert bumped is not snap
    # Shared array references are fine — frozen semantics live at the
    # dataclass level, not the leaf level. JAX will treat leaves as
    # immutable by convention.
    assert bumped.hosts is snap.hosts


@pytest.mark.integration
def test_round_trip_preserves_host_observable_state(global_state) -> None:
    """legacy -> EnvState -> legacy must preserve every field the env reads."""
    snap = from_global_state(global_state, agent_ids=AGENTS)
    rebuilt = to_global_state(snap)

    assert sorted(rebuilt.all_hosts.keys()) == sorted(global_state.all_hosts.keys())

    for ip in global_state.all_hosts:
        original = global_state.all_hosts[ip]
        recon = rebuilt.all_hosts[ip]

        assert recon.hostname == original.hostname
        assert recon.subnet_cidr == original.subnet_cidr
        assert recon.status == original.status
        assert recon.privilege == original.privilege
        assert recon.edr_active == original.edr_active
        assert recon.is_domain_controller == original.is_domain_controller
        assert recon.contains_honeytokens == getattr(
            original, 'contains_honeytokens', False
        )
        assert recon.os == original.os
        assert list(recon.services) == list(original.services)
        assert list(recon.vulnerabilities) == list(original.vulnerabilities)
        assert list(recon.cached_credentials) == list(original.cached_credentials)
        assert list(recon.system_tokens) == list(original.system_tokens)
        assert recon.compromised_by == original.compromised_by
        assert pytest.approx(recon.human_vulnerability_score, abs=1e-6) == (
            original.human_vulnerability_score
        )

        # Decoy field is encoded via DECOY_CODES; values outside the codebook
        # collapse to 'inactive'. Document that contract here.
        if original.decoy in DECOY_CODES:
            assert recon.decoy == original.decoy
        else:
            assert recon.decoy == 'inactive'


@pytest.mark.integration
def test_round_trip_preserves_agent_budgets_and_knowledge(env_sim) -> None:
    legacy = env_sim.global_state
    snap = from_global_state(legacy, agent_ids=tuple(env_sim.possible_agents))
    rebuilt = to_global_state(snap)

    for agent in env_sim.possible_agents:
        assert rebuilt.agent_energy[agent] == legacy.agent_energy[agent]
        assert rebuilt.agent_funds[agent] == legacy.agent_funds[agent]
        assert rebuilt.agent_compute[agent] == legacy.agent_compute[agent]
        assert rebuilt.agent_locked_until.get(agent, 0) == legacy.agent_locked_until.get(
            agent, 0
        )
        assert rebuilt.agent_knowledge.get(agent, set()) == legacy.agent_knowledge.get(
            agent, set()
        )
        assert rebuilt.agent_inventory.get(agent, set()) == legacy.agent_inventory.get(
            agent, set()
        )


@pytest.mark.fast
def test_codebooks_are_stable() -> None:
    """Codebook ordering is load-bearing — pin it explicitly."""
    assert STATUS_CODES == ('online', 'isolated', 'kernel_panic')
    assert PRIVILEGE_CODES == ('None', 'User', 'Root')
    assert DECOY_CODES[:2] == ('inactive', 'active')


@pytest.mark.fast
def test_unknown_host_count_rejected() -> None:
    from netforge_rl.core.state import GlobalNetworkState, Host

    too_small = GlobalNetworkState()
    too_small.register_host(Host('1.1.1.1', 'a', '1.1.1.0/24'))
    with pytest.raises(ValueError, match='Expected exactly'):
        from_global_state(too_small, agent_ids=AGENTS)
