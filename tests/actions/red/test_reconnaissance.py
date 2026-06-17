import pytest
from netforge_rl.actions.red.reconnaissance import NetworkScan, DiscoverRemoteSystems, DiscoverNetworkServices
from netforge_rl.core.state import GlobalNetworkState, Host

@pytest.fixture
def red_agent():
    return 'red_operator'

@pytest.mark.fast
def test_network_scan_execution(red_agent):
    state = GlobalNetworkState()
    target_subnet = '192.168.1.0/24'
    action = NetworkScan(agent_id=red_agent, target_subnet=target_subnet)
    result = action.execute(state)
    assert result.success is True
    assert result.observation_data['discovered_subnet'] == target_subnet

@pytest.mark.fast
def test_discover_remote_systems_execution(red_agent):
    state = GlobalNetworkState()
    target_subnet = '192.168.1.0/24'
    action = DiscoverRemoteSystems(agent_id=red_agent, target_subnet=target_subnet)
    state.register_host(Host(ip='192.168.1.10', hostname='WebSrv', subnet_cidr=target_subnet))
    result = action.execute(state)
    assert result.success is True
    assert 'hosts' in result.observation_data
    assert '192.168.1.10' in result.observation_data['hosts']

@pytest.mark.fast
def test_discover_network_services_execution(red_agent):
    state = GlobalNetworkState()
    target_ip = '192.168.1.10'
    host = Host(ip=target_ip, hostname='WebSrv', subnet_cidr='192.168.1.0/24')
    host.services = ['HTTP', 'SSH']
    state.register_host(host)
    action = DiscoverNetworkServices(agent_id=red_agent, target_ip=target_ip)
    result = action.execute(state)
    assert result.success is True
    assert 'HTTP' in result.observation_data['services']

@pytest.mark.fast
def test_discover_remote_systems_decoys(red_agent):
    state = GlobalNetworkState()
    target_subnet = '192.168.1.0/24'
    action = DiscoverRemoteSystems(agent_id=red_agent, target_subnet=target_subnet)
    host = Host(ip='192.168.1.50', hostname='Honeypot', subnet_cidr=target_subnet)
    host.decoy = 'active'
    state.register_host(host)
    result = action.execute(state)
    assert result.success is True
    assert '10.x.x.99' in result.observation_data['hosts']

@pytest.mark.fast
def test_discover_network_services_decoys(red_agent):
    state = GlobalNetworkState()
    target_ip = '192.168.1.100'
    host = Host(ip=target_ip, hostname='Honeypot', subnet_cidr='192.168.1.0/24')
    host.decoy = 'Apache'
    state.register_host(host)
    action = DiscoverNetworkServices(agent_id=red_agent, target_ip=target_ip)
    result = action.execute(state)
    assert 'Fake_Apache_80' in result.observation_data['services']