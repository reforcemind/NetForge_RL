import pytest
from netforge_rl.core.state import Host, Firewall

@pytest.mark.fast
def test_host_initialization():
    host = Host(ip='10.0.1.5', hostname='TestNode', subnet_cidr='10.0.1.0/24')
    assert host.ip == '10.0.1.5'
    assert host.subnet_cidr == '10.0.1.0/24'
    assert host.privilege == 'None'
    assert host.compromised_by == 'None'
    assert host.status == 'online'
    assert isinstance(host.services, list)
    assert isinstance(host.system_tokens, list)

@pytest.mark.fast
def test_global_state_generation(global_state):
    assert len(global_state.subnets) > 0
    assert len(global_state.all_hosts) > 0
    subnets = [s.cidr for s in global_state.subnets.values()]
    assert '192.168.1.0/24' in subnets
    assert '10.0.0.0/24' in subnets
    assert '10.0.1.0/24' in subnets

@pytest.mark.fast
def test_ztna_routing_unauthenticated(global_state):
    secure_host_ip = next((ip for ip, h in global_state.all_hosts.items() if h.subnet_cidr == '10.0.1.0/24'))
    assert global_state.can_route_to(secure_host_ip, agent_id='red_operator') is False

@pytest.mark.fast
def test_ztna_routing_pivot_requirements(global_state):
    corp_ip = next((ip for ip, h in global_state.all_hosts.items() if h.subnet_cidr == '10.0.0.0/24'))
    secure_ip = next((ip for ip, h in global_state.all_hosts.items() if h.subnet_cidr == '10.0.1.0/24'))
    assert global_state.can_route_to(corp_ip, agent_id='red_operator') is False
    assert global_state.can_route_to(secure_ip, agent_id='red_operator') is False
    dmz_ip = next((ip for ip, h in global_state.all_hosts.items() if h.subnet_cidr == '192.168.1.0/24'))
    global_state.all_hosts[dmz_ip].privilege = 'Root'
    assert global_state.can_route_to(corp_ip, agent_id='red_operator') is True
    assert global_state.can_route_to(secure_ip, agent_id='red_operator') is False
    global_state.all_hosts[corp_ip].privilege = 'Root'
    global_state.agent_inventory['red_operator'] = {'Enterprise_Admin_Token'}
    assert global_state.can_route_to(secure_ip, agent_id='red_operator') is True

@pytest.mark.fast
def test_firewall_blocking(global_state):
    dmz_ip = next((ip for ip, h in global_state.all_hosts.items() if h.subnet_cidr == '192.168.1.0/24'))
    port = 80
    assert global_state.can_route_to(dmz_ip, port=port) is True
    global_state.firewalls['global'] = Firewall('global')
    global_state.firewalls['global'].block_port('192.168.1.0/24', port)
    assert global_state.can_route_to(dmz_ip, port=port) is False

@pytest.mark.fast
def test_isolated_host_unreachable(global_state):
    dmz_ip = next((ip for ip, h in global_state.all_hosts.items() if h.subnet_cidr == '192.168.1.0/24'))
    global_state.all_hosts[dmz_ip].status = 'isolated'
    assert global_state.can_route_to(dmz_ip, agent_id='red_operator') is False
    assert global_state.can_route_to(dmz_ip, agent_id='blue_operator') is False