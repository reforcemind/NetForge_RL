"""Parity tests: pure ``apply_state_delta`` vs legacy ``GlobalNetworkState.apply_delta``.

For every supported delta key, we drive both interpreters with the same input
and assert observable equivalence after a legacy <-> EnvState round-trip.
This is the contract slice 4 will rely on when wiring ParallelEnv onto the
functional core: the legacy step loop can emit the same state_deltas dict,
and either interpreter will produce the same downstream state.
"""

import pytest

from netforge_rl.core.functional import (
    apply_state_delta,
    apply_state_deltas,
    from_global_state,
    to_global_state,
)


AGENTS = (
    'red_operator',
    'blue_dmz',
    'blue_internal',
    'blue_restricted',
)


def _first_ip(state) -> str:
    return sorted(state.all_hosts.keys())[0]


@pytest.mark.fast
def test_host_status_delta_parity(global_state) -> None:
    ip = _first_ip(global_state)
    key, val = f'hosts/{ip}/status', 'isolated'

    snap = from_global_state(global_state, agent_ids=AGENTS)
    new_snap = apply_state_delta(snap, key, val)
    global_state.apply_delta(key, val)

    rebuilt = to_global_state(new_snap)
    assert rebuilt.all_hosts[ip].status == 'isolated'
    assert global_state.all_hosts[ip].status == 'isolated'


@pytest.mark.fast
def test_host_kernel_panic_status_delta_parity(global_state) -> None:
    ip = _first_ip(global_state)
    snap = from_global_state(global_state, agent_ids=AGENTS)
    new_snap = apply_state_delta(snap, f'hosts/{ip}/status', 'kernel_panic')
    assert to_global_state(new_snap).all_hosts[ip].status == 'kernel_panic'


@pytest.mark.fast
def test_host_privilege_delta_parity(global_state) -> None:
    ip = _first_ip(global_state)
    snap = from_global_state(global_state, agent_ids=AGENTS)

    rebuilt = to_global_state(
        apply_state_delta(snap, f'hosts/{ip}/privilege', 'Root')
    )
    global_state.apply_delta(f'hosts/{ip}/privilege', 'Root')

    assert rebuilt.all_hosts[ip].privilege == 'Root'
    assert global_state.all_hosts[ip].privilege == 'Root'


@pytest.mark.fast
def test_host_compromised_by_delta_parity(global_state) -> None:
    ip = _first_ip(global_state)
    snap = from_global_state(global_state, agent_ids=AGENTS)

    rebuilt = to_global_state(
        apply_state_delta(snap, f'hosts/{ip}/compromised_by', 'red_operator')
    )
    global_state.apply_delta(f'hosts/{ip}/compromised_by', 'red_operator')

    assert rebuilt.all_hosts[ip].compromised_by == 'red_operator'
    assert global_state.all_hosts[ip].compromised_by == 'red_operator'


@pytest.mark.fast
def test_host_edr_active_delta_parity(global_state) -> None:
    ip = _first_ip(global_state)
    snap = from_global_state(global_state, agent_ids=AGENTS)

    rebuilt = to_global_state(
        apply_state_delta(snap, f'hosts/{ip}/edr_active', True)
    )
    global_state.apply_delta(f'hosts/{ip}/edr_active', True)

    assert rebuilt.all_hosts[ip].edr_active is True
    assert global_state.all_hosts[ip].edr_active is True


@pytest.mark.fast
def test_host_os_meta_delta(global_state) -> None:
    ip = _first_ip(global_state)
    snap = from_global_state(global_state, agent_ids=AGENTS)

    rebuilt = to_global_state(apply_state_delta(snap, f'hosts/{ip}/os', 'Linux_Arch'))
    assert rebuilt.all_hosts[ip].os == 'Linux_Arch'


@pytest.mark.fast
def test_knowledge_delta_parity(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    ip = '10.99.99.99'

    rebuilt = to_global_state(
        apply_state_delta(snap, f'knowledge/red_operator/{ip}')
    )
    global_state.apply_delta(f'knowledge/red_operator/{ip}')

    assert ip in rebuilt.agent_knowledge['red_operator']
    assert ip in global_state.agent_knowledge['red_operator']


@pytest.mark.fast
def test_knowledge_delta_unknown_agent_ignored(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    out = apply_state_delta(snap, 'knowledge/ghost_agent/1.2.3.4')
    assert out is snap or out.knowledge == snap.knowledge


@pytest.mark.fast
def test_unknown_delta_is_noop(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)
    out = apply_state_delta(snap, 'firewall/block/1.0.0.0_slash_24/22')
    assert out is snap  # short-circuit, same instance


@pytest.mark.fast
def test_unknown_host_attribute_is_noop(global_state) -> None:
    ip = _first_ip(global_state)
    snap = from_global_state(global_state, agent_ids=AGENTS)
    out = apply_state_delta(snap, f'hosts/{ip}/nonexistent_attr', 42)
    assert out is snap


@pytest.mark.fast
def test_command_object_delta_is_noop(global_state) -> None:
    snap = from_global_state(global_state, agent_ids=AGENTS)

    class FakeCommand:
        def execute(self, state):
            raise AssertionError('should not be called by functional interpreter')

    out = apply_state_delta(snap, FakeCommand())
    assert out is snap


@pytest.mark.fast
def test_original_state_is_not_mutated(global_state) -> None:
    ip = _first_ip(global_state)
    snap = from_global_state(global_state, agent_ids=AGENTS)
    original_status = snap.hosts.status[snap.host_index(ip)]
    _ = apply_state_delta(snap, f'hosts/{ip}/status', 'isolated')
    assert snap.hosts.status[snap.host_index(ip)] == original_status


@pytest.mark.fast
def test_apply_state_deltas_dict(global_state) -> None:
    ip = _first_ip(global_state)
    snap = from_global_state(global_state, agent_ids=AGENTS)
    deltas = {
        f'hosts/{ip}/status': 'isolated',
        f'hosts/{ip}/privilege': 'Root',
        f'hosts/{ip}/edr_active': True,
    }
    out = apply_state_deltas(snap, deltas)
    legacy = to_global_state(out).all_hosts[ip]
    assert legacy.status == 'isolated'
    assert legacy.privilege == 'Root'
    assert legacy.edr_active is True
