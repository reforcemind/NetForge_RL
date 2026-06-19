import pytest
import numpy as np
from netforge_rl.core.observation import BaseObservation
from netforge_rl.core.state import GlobalNetworkState, Host


@pytest.mark.fast
def test_observation_update_red(red_agent_id):
    obs = BaseObservation(red_agent_id)
    state = GlobalNetworkState()
    target_ip = '10.0.0.5'
    host = Host(ip=target_ip, hostname='Target', subnet_cidr='10.0.0.0/24')
    host.privilege = 'Root'
    state.register_host(host)
    state.update_knowledge(red_agent_id, target_ip)
    obs.update_from_state(state, [])
    assert target_ip in obs.visible_hosts
    assert obs.visible_hosts[target_ip]['state'] == 'compromised'
    assert obs.visible_hosts[target_ip]['decoy'] == 'unknown'


@pytest.mark.fast
def test_observation_update_blue(blue_agent_id):
    obs = BaseObservation(blue_agent_id)
    state = GlobalNetworkState()
    target_ip = '10.0.0.10'
    host = Host(ip=target_ip, hostname='Target', subnet_cidr='10.0.0.0/24')
    host.status = 'isolated'
    state.register_host(host)
    state.update_knowledge(blue_agent_id, target_ip)
    obs.update_from_state(state, [])
    assert target_ip in obs.visible_hosts
    assert obs.visible_hosts[target_ip]['state'] == 'unknown'
    assert obs.visible_hosts[target_ip]['status'] == 'isolated'


@pytest.mark.fast
def test_observation_siem_alerts(blue_agent_id):
    obs = BaseObservation(blue_agent_id)
    state = GlobalNetworkState()
    state.current_tick = 5
    state.siem_log_buffer.append({'arrival_tick': 2, 'msg': 'Detection A'})
    state.siem_log_buffer.append({'arrival_tick': 10, 'msg': 'Detection B'})
    obs.update_from_state(state, [])
    assert len(obs.siem_alerts) == 1
    assert obs.siem_alerts[0]['msg'] == 'Detection A'


@pytest.mark.fast
def test_observation_to_numpy_serialization():
    obs = BaseObservation('red_operator_0')
    obs.network_telemetry = {
        'global_alert_level': 0.75,
        'total_isolated_subnets': 2,
        'active_alerts': 5,
    }
    obs.objective_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    obs.visible_hosts['10.0.0.5'] = {'state': 'compromised'}
    vec = obs.to_numpy(max_size=32)
    assert vec.shape == (32,)
    assert vec[0] == 0.75
    assert vec[1] == 0.2
    assert vec[2] == 0.25
    assert np.allclose(vec[3:8], obs.objective_vector)
    assert vec[8] == pytest.approx(5 / 255.0)
    assert vec[9] == 1.0
