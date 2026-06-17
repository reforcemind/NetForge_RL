import pytest
from netforge_rl.docker_bridge.hypervisor_base import HypervisorResult
from netforge_rl.docker_bridge.mock_hypervisor import MockHypervisor

@pytest.fixture
def mock_hvr():
    return MockHypervisor(seed=42)

@pytest.mark.fast
def test_mock_hypervisor_dispatch(mock_hvr):
    result = mock_hvr.dispatch('ExploitEternalBlue', '10.0.1.5', 'Windows_Server_2016')
    assert isinstance(result, HypervisorResult)
    assert result.action_name == 'ExploitEternalBlue'
    assert result.latency_ms > 0
    assert result.success is True
    assert result.return_code == 0

@pytest.mark.fast
def test_mock_hypervisor_os_penalty(mock_hvr):
    result = mock_hvr.dispatch('ExploitEternalBlue', '10.0.1.5', 'Linux_Ubuntu')
    assert result.success is False
    assert result.return_code == 1

@pytest.mark.fast
def test_mock_hypervisor_unknown_action(mock_hvr):
    result = mock_hvr.dispatch('UnknownAction', '10.0.0.1', 'Windows')
    assert result.success is False
    assert 'UnknownAction failed' in result.stdout
    assert result.return_code == 1