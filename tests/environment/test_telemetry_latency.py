import numpy as np
import pytest

from netforge_rl.environment.parallel_env import NetForgeRLEnv


def _step(env, rng):
    return env.step(
        {
            a: np.array([rng.integers(0, n) for n in env.action_space(a).nvec])
            for a in env.agents
        }
    )


@pytest.mark.fast
def test_latency_delays_log_visibility():
    """With latency > 0, logs generated on a tick are not yet in the buffer."""
    immediate = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 10})
    immediate.reset(seed=1)
    delayed = NetForgeRLEnv(
        {'scenario_type': 'ransomware', 'max_ticks': 10, 'log_latency': 5}
    )
    delayed.reset(seed=1)

    _step(immediate, np.random.default_rng(1))
    _step(delayed, np.random.default_rng(1))

    assert len(delayed.global_state.siem_log_buffer) < len(
        immediate.global_state.siem_log_buffer
    )


@pytest.mark.fast
def test_delayed_logs_eventually_arrive():
    env = NetForgeRLEnv(
        {'scenario_type': 'ransomware', 'max_ticks': 20, 'log_latency': 3}
    )
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(8):
        if not env.agents:
            break
        _step(env, rng)
    assert len(env.global_state.siem_log_buffer) > 0


@pytest.mark.fast
def test_dhcp_interval_zero_disables_reallocation():
    env = NetForgeRLEnv(
        {'scenario_type': 'ransomware', 'max_ticks': 60, 'dhcp_interval': 0}
    )
    env.reset(seed=1)
    before = set(env.global_state.all_hosts.keys())
    rng = np.random.default_rng(1)
    for _ in range(50):
        if not env.agents:
            break
        _step(env, rng)
    # No DHCP churn means the host address set is unchanged by reallocation.
    assert set(env.global_state.all_hosts.keys()) == before
