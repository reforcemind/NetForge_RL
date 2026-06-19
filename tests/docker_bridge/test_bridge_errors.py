import pytest
from unittest.mock import MagicMock, patch
from netforge_rl.docker_bridge.bridge import DockerBridge
from netforge_rl.docker_bridge.mock_hypervisor import MockHypervisor


@pytest.mark.fast
def test_bridge_fallback_on_docker_failure():
    with patch(
        'netforge_rl.docker_bridge.docker_hypervisor.DockerHypervisor.is_available',
        return_value=False,
    ):
        bridge = DockerBridge(mode='real')
        assert isinstance(bridge._driver, MockHypervisor)
        assert bridge.mode == 'real'


@pytest.mark.fast
def test_bridge_reward_mapping_success():
    bridge = DockerBridge(mode='sim')
    mock_result = MagicMock()
    mock_result.success = True
    assert bridge.reward_delta(mock_result) == 5.0


@pytest.mark.fast
def test_bridge_reward_mapping_noisy_failure():
    bridge = DockerBridge(mode='sim')
    mock_result = MagicMock()
    mock_result.success = False
    mock_result.return_code = 1
    mock_result.latency_ms = 6000.0
    assert bridge.reward_delta(mock_result) == -20.0


@pytest.mark.fast
def test_bridge_reward_mapping_infra_error():
    bridge = DockerBridge(mode='sim')
    mock_result = MagicMock()
    mock_result.success = False
    mock_result.return_code = 2
    assert bridge.reward_delta(mock_result) == -25.0
