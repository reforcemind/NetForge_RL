import pytest
from netforge_rl.agents.green_agent import GreenAgent

@pytest.fixture
def green_agent():
    return GreenAgent()

@pytest.mark.fast
def test_green_agent_generate_noise_day(green_agent, global_state):
    noise = green_agent.generate_noise(0, global_state)
    assert 'alerts' in noise
    for alert in noise['alerts']:
        assert 'type' in alert
        assert 'severity' in alert

@pytest.mark.fast
def test_green_agent_generate_noise_night(green_agent, global_state):
    noise = green_agent.generate_noise(110, global_state)
    assert 'alerts' in noise
    for alert in noise['alerts']:
        assert 'type' in alert

@pytest.mark.fast
def test_green_agent_empty_hosts(green_agent):
    mock_state = type('MockState', (), {'all_hosts': {}})()
    noise = green_agent.generate_noise(0, mock_state)
    assert noise == {'alerts': []}

@pytest.mark.fast
def test_green_agent_cycle_positions(green_agent, global_state):
    noise_day = green_agent.generate_noise(100, global_state)
    noise_night = green_agent.generate_noise(101, global_state)
    assert isinstance(noise_day['alerts'], list)
    assert isinstance(noise_night['alerts'], list)