"""Smoke + reward-sign tests for the IoT grid + OT Stuxnet scenarios."""

import numpy as np
import pytest

from netforge_rl.environment.parallel_env import NetForgeRLEnv
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
