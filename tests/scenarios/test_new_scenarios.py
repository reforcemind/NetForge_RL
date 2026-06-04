import numpy as np
import pytest

from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.scenarios.cloud_hybrid import CloudHybridScenario
from netforge_rl.scenarios.iot_grid import IoTGridScenario
from netforge_rl.scenarios.ot_stuxnet import OTStuxnetScenario


@pytest.mark.fast
def test_env_resolves_iot_grid_scenario_type() -> None:
    env = NetForgeRLEnv({'scenario_type': 'iot_grid', 'max_ticks': 5})
    assert isinstance(env.scenario, IoTGridScenario)


@pytest.mark.fast
def test_env_resolves_ot_stuxnet_scenario_type() -> None:
    env = NetForgeRLEnv({'scenario_type': 'ot_stuxnet', 'max_ticks': 5})
    assert isinstance(env.scenario, OTStuxnetScenario)


@pytest.mark.integration
@pytest.mark.parametrize('scenario_type', ['iot_grid', 'ot_stuxnet'])
def test_episode_runs_under_new_scenarios(scenario_type) -> None:
    env = NetForgeRLEnv({'scenario_type': scenario_type, 'max_ticks': 5})
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    steps = 0
    while env.agents and steps < 5:
        acts = {
            a: np.array([rng.integers(0, n) for n in env.action_space(a).nvec], dtype=np.int64)
            for a in env.agents
        }
        env.step(acts)
        steps += 1
    assert steps > 0


@pytest.mark.fast
def test_iot_grid_blue_penalised_when_controller_compromised() -> None:
    env = NetForgeRLEnv({'scenario_type': 'iot_grid', 'max_ticks': 5})
    env.reset(seed=0)
    dc = next(h for h in env.global_state.all_hosts.values() if h.is_domain_controller)
    dc.compromised_by = 'red_operator'
    r_bad = env.scenario.calculate_reward('blue_dmz', env.global_state)
    dc.compromised_by = 'None'
    r_good = env.scenario.calculate_reward('blue_dmz', env.global_state)
    assert r_good > r_bad


@pytest.mark.fast
def test_ot_stuxnet_terminates_on_kinetic_destruction() -> None:
    env = NetForgeRLEnv({'scenario_type': 'ot_stuxnet', 'max_ticks': 5})
    env.reset(seed=0)
    h = next(iter(env.global_state.all_hosts.values()))
    h.system_integrity = 'kinetic_destruction'
    term = env.scenario.check_termination(env.global_state)
    assert all(term.values())


@pytest.mark.fast
def test_env_resolves_cloud_hybrid_scenario_type():
    env = NetForgeRLEnv({'scenario_type': 'cloud_hybrid', 'max_ticks': 5})
    assert isinstance(env.scenario, CloudHybridScenario)


@pytest.mark.fast
def test_cloud_hybrid_red_rewarded_for_secure_breach():
    env = NetForgeRLEnv({'scenario_type': 'cloud_hybrid', 'max_ticks': 5})
    env.reset(seed=0)
    secure = next(
        (h for h in env.global_state.all_hosts.values()
         if h.subnet_cidr == '10.0.1.0/24'),
        None,
    )
    if secure is None:
        pytest.skip('topology had no Secure subnet host')
    from netforge_rl.core.action import ActionEffect
    eff = ActionEffect(
        success=True,
        state_deltas={f'hosts/{secure.ip}/privilege': 'User'},
        observation_data={},
    )
    r = env.scenario.calculate_reward('red_operator', env.global_state, eff)
    assert r >= CloudHybridScenario.SECURE_BREACH_REWARD - 1.0


@pytest.mark.fast
def test_cloud_hybrid_terminates_when_secure_fully_owned():
    env = NetForgeRLEnv({'scenario_type': 'cloud_hybrid', 'max_ticks': 5})
    env.reset(seed=0)
    secure = [
        h for h in env.global_state.all_hosts.values()
        if h.subnet_cidr == '10.0.1.0/24'
    ]
    if not secure:
        pytest.skip('topology had no Secure subnet hosts')
    for h in secure:
        h.compromised_by = 'red_operator'
    term = env.scenario.check_termination(env.global_state)
    assert all(term.values())


@pytest.mark.integration
def test_episode_runs_under_cloud_hybrid():
    env = NetForgeRLEnv({'scenario_type': 'cloud_hybrid', 'max_ticks': 5})
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        if not env.agents:
            break
        acts = {
            a: np.array([rng.integers(0, n) for n in env.action_space(a).nvec],
                        dtype=np.int64)
            for a in env.agents
        }
        env.step(acts)
