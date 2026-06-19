import pytest
from netforge_rl.docker_bridge.bridge import DockerBridge
from netforge_rl.docker_bridge.hypervisor_base import HypervisorResult


@pytest.fixture
def bridge():
    return DockerBridge(mode='sim')


@pytest.mark.fast
def test_bridge_mode_switching(bridge):
    assert bridge.mode == 'sim'
    real_bridge = DockerBridge(mode='real')
    assert real_bridge.mode == 'real'


@pytest.mark.fast
def test_bridge_dispatch_routing(bridge):
    result = bridge.dispatch('ExploitEternalBlue', '10.0.1.5', 'Windows_7')
    assert isinstance(result, HypervisorResult)
    assert result.action_name == 'ExploitEternalBlue'


@pytest.mark.fast
def test_bridge_reward_delta(bridge):
    res_suc = HypervisorResult(True, '', 0, 100.0, 'Act', '1.1.1.1', 'Win', 'mock')
    assert bridge.reward_delta(res_suc) == 5.0
    res_fail = HypervisorResult(False, '', 1, 100.0, 'Act', '1.1.1.1', 'Win', 'mock')
    assert bridge.reward_delta(res_fail) == -10.0
    res_noisy = HypervisorResult(False, '', 1, 6000.0, 'Act', '1.1.1.1', 'Win', 'mock')
    assert bridge.reward_delta(res_noisy) == -20.0
    res_err = HypervisorResult(False, '', 2, 100.0, 'Act', '1.1.1.1', 'Win', 'mock')
    assert bridge.reward_delta(res_err) == -25.0
