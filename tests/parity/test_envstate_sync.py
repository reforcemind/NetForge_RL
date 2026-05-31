"""End-to-end converter exercise: legacy step loop ↔ frozen EnvState.

Runs a deterministic episode through the legacy backend and, at every step,
materializes a frozen :class:`EnvState` via ``env.to_envstate()``. We assert
the snapshot's vectorized host arrays agree with the legacy ``GlobalNetworkState``
on the fields that the functional core currently tracks.

This is the canary that catches drift between the two representations as
later slices (state mutations from new action types, new categorical
values) land — it's load-bearing for the JAX backend migration.
"""

import numpy as np
import pytest

from netforge_rl.core.functional import (
    DECOY_CODES,
    PRIVILEGE_CODES,
    STATUS_CODES,
)
from netforge_rl.environment.parallel_env import NetForgeRLEnv


SCENARIO_CONFIG = {
    'scenario_type': 'ransomware',
    'sim2real_mode': 'sim',
    'nlp_backend': 'tfidf',
    'max_ticks': 30,
    'log_latency': 2,
}


def _scripted_actions(env, rng):
    actions = {}
    for agent in env.agents:
        space = env.action_space(agent)
        actions[agent] = np.array(
            [rng.integers(0, n) for n in space.nvec], dtype=np.int64
        )
    return actions


def _assert_snapshot_matches_legacy(env) -> None:
    snap = env.to_envstate()
    legacy = env.global_state

    sorted_ips = sorted(legacy.all_hosts.keys())
    assert list(snap.meta.ip) == sorted_ips

    for i, ip in enumerate(sorted_ips):
        host = legacy.all_hosts[ip]
        # Status: tolerate unknown statuses by collapsing both sides to code 0
        # (the legacy decoder behavior already documented in test_functional).
        expected_status = (
            STATUS_CODES.index(host.status) if host.status in STATUS_CODES else 0
        )
        assert int(snap.hosts.status[i]) == expected_status, (
            f'status drift @{ip}: snap={snap.hosts.status[i]} legacy={host.status}'
        )
        assert int(snap.hosts.privilege[i]) == PRIVILEGE_CODES.index(host.privilege)

        if host.decoy in DECOY_CODES:
            assert int(snap.hosts.decoy[i]) == DECOY_CODES.index(host.decoy)
        # else: silently collapsed to 0 (inactive) — see DECOY_CODES contract.

        assert bool(snap.hosts.edr_active[i]) == bool(host.edr_active)
        assert bool(snap.hosts.is_domain_controller[i]) == bool(
            host.is_domain_controller
        )

    for j, agent in enumerate(env.possible_agents):
        assert int(snap.agent_energy[j]) == legacy.agent_energy.get(agent, 0)
        assert int(snap.agent_funds[j]) == legacy.agent_funds.get(agent, 0)
        assert int(snap.agent_locked_until[j]) == legacy.agent_locked_until.get(
            agent, 0
        )

    assert int(snap.current_tick) == int(legacy.current_tick)


@pytest.mark.integration
def test_envstate_snapshot_matches_legacy_at_reset() -> None:
    env = NetForgeRLEnv(SCENARIO_CONFIG)
    env.reset(seed=42)
    _assert_snapshot_matches_legacy(env)


@pytest.mark.integration
def test_envstate_snapshot_matches_legacy_across_episode() -> None:
    env = NetForgeRLEnv(SCENARIO_CONFIG)
    env.reset(seed=42)
    rng = np.random.default_rng(42)

    for _ in range(20):
        if not env.agents:
            break
        env.step(_scripted_actions(env, rng))
        _assert_snapshot_matches_legacy(env)


@pytest.mark.fast
def test_snapshot_is_frozen() -> None:
    import dataclasses

    env = NetForgeRLEnv(SCENARIO_CONFIG)
    env.reset(seed=42)
    snap = env.to_envstate()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.current_tick = 999  # type: ignore[misc]
