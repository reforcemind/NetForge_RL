from netforge_rl.agents.rule_based_blue import RuleBasedBlueAgent
from netforge_rl.core.state import GlobalNetworkState, Host
from netforge_rl.actions.blue.mitigation import IsolateHost, RestoreHost
from netforge_rl.actions.blue.analysis import Analyze, Monitor
from netforge_rl.actions.blue.deception import DeployHoneytoken
from netforge_rl.actions.blue.edr import DeployEDR


def test_rule_based_blue_agent_reset():
    agent = RuleBasedBlueAgent('blue')
    agent._isolated.add('192.168.1.1')
    agent._honeytokened.add('192.168.1.1')
    agent._edr_deployed.add('192.168.1.1')
    agent.reset()
    assert len(agent._isolated) == 0
    assert len(agent._honeytokened) == 0
    assert len(agent._edr_deployed) == 0


def test_rule_based_blue_agent_get_action():
    agent = RuleBasedBlueAgent('blue')
    state = GlobalNetworkState()

    action = agent.get_action(None, state)
    assert action is None

    host1 = Host('192.168.1.1', 'h1', '192.168.1.0/24')
    host1.privilege = 'User'
    host1.status = 'online'
    state.register_host(host1)

    action = agent.get_action(None, state)
    assert isinstance(action, IsolateHost)
    assert action.target_ip == '192.168.1.1'

    host1.privilege = 'None'
    host1.edr_active = True
    agent.reset()

    action = agent.get_action(None, state)
    assert isinstance(action, Analyze)
    assert action.target_ip == '192.168.1.1'

    host1.edr_active = False
    host1.cvss_score = 5.0
    host2 = Host('192.168.1.2', 'h2', '192.168.1.0/24')
    host2.cvss_score = 9.0
    host2.status = 'online'
    state.register_host(host2)

    action = agent.get_action(None, state)
    assert isinstance(action, DeployEDR)
    assert action.target_ip == '192.168.1.2'

    agent._edr_deployed.add('192.168.1.1')
    agent._edr_deployed.add('192.168.1.2')

    action = agent.get_action(None, state)
    assert isinstance(action, DeployHoneytoken)
    assert action.target_ip == '192.168.1.2'

    host3 = Host('192.168.1.3', 'h3', '192.168.1.0/24')
    host3.status = 'isolated'
    host3.compromised_by = 'None'
    state.register_host(host3)

    agent._honeytokened.add('192.168.1.1')
    agent._honeytokened.add('192.168.1.2')

    action = agent.get_action(None, state)
    assert isinstance(action, RestoreHost)
    assert action.target_ip == '192.168.1.3'

    host3.status = 'online'
    agent._edr_deployed.add('192.168.1.3')
    agent._honeytokened.add('192.168.1.3')

    action = agent.get_action(None, state)
    assert isinstance(action, Monitor)
