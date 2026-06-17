import numpy as np
from netforge_rl.environment.parallel_env import NetForgeRLEnv

def test_asynchronous_step_delta_t():
    env = NetForgeRLEnv(scenario_config={'scenario_type': 'apt_espionage'})
    obs, infos = env.reset()
    from netforge_rl.core.action import BaseAction, ActionEffect

    class MockLongAction(BaseAction):

        def __init__(self, agent_id):
            super().__init__(agent_id)
            self.duration = 10
            self.cost = 0

        def validate(self, state):
            return True

        def execute(self, state):
            return ActionEffect(True, {}, {})
    env.event_queue.append({'completion_tick': 10, 'agent': 'blue_dmz', 'action': MockLongAction('blue_dmz'), 'effect': ActionEffect(True, {}, {}), 'target_ip': '10.0.0.1'})
    obs, rewards, terminate, truncate, infos = env.step({})
    assert env.current_tick == 10
    for agent in env.possible_agents:
        assert 'delta_t' in infos[agent]
        assert infos[agent]['delta_t'] == 10.0
        assert 'delta_t_norm' in infos[agent]
        assert np.isclose(infos[agent]['delta_t_norm'], 0.2)
        assert 'delta_t' in obs[agent]
        assert np.isclose(obs[agent]['delta_t'][0], 0.2)

def test_subnet_filtering():
    env = NetForgeRLEnv(scenario_config={'scenario_type': 'apt_espionage'})
    env.reset()
    env.siem_logger._push_to_buffer('DMZ_ALERT_XML', '192.168.1.0/24', env.global_state)
    env.siem_logger._push_to_buffer('INTERNAL_ALERT_XML', '10.0.0.0/24', env.global_state)
    obs, rewards, terminate, truncate, infos = env.step({})
    dmz_logs = env.siem_logger.get_filtered_logs(env.global_state, subnet_tag='dmz')
    assert 'DMZ_ALERT_XML' in dmz_logs
    assert 'INTERNAL_ALERT_XML' not in dmz_logs
    internal_logs = env.siem_logger.get_filtered_logs(env.global_state, subnet_tag='internal')
    assert 'INTERNAL_ALERT_XML' in internal_logs
    assert 'DMZ_ALERT_XML' not in internal_logs

def test_green_agent_xml_fidelity():
    from netforge_rl.agents.green_agent import GreenAgent
    from netforge_rl.core.state import GlobalNetworkState, Host, Subnet
    state = GlobalNetworkState()
    subnet = Subnet('192.168.1.0/24', 'DMZ')
    state.add_subnet(subnet)
    host = Host('192.168.1.1', 'Target', '192.168.1.0/24')
    state.register_host(host)
    ga = GreenAgent()
    import random
    random.seed(42)
    found_xml = False
    for _ in range(100):
        noise = ga.generate_noise(1, state)
        for alert in noise['alerts']:
            if '<Event' in alert['data'] and 'xmlns' in alert['data']:
                found_xml = True
                break
        if found_xml:
            break
    assert found_xml, 'GreenAgent should have produced at least one XML encoded log.'