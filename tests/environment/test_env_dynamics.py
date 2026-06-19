import pytest
from unittest.mock import patch
from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.actions.red.exploits import ExploitEternalBlue
from netforge_rl.core.action import ActionEffect


@pytest.fixture
def env(env_config):
    env = NetForgeRLEnv(env_config)
    env.reset(seed=42)
    return env


class MagicMockAction:
    def __init__(self, cost=1, duration=1):
        self.cost = cost
        self.duration = duration
        self.target_ip = '1.2.3.4'

    def validate(self, state):
        return True

    def execute(self, state):
        return ActionEffect(success=True, state_deltas={}, observation_data={})


@pytest.mark.fast
def test_soc_budget_per_agent_not_global(env):
    env.reset(seed=42)
    agent = 'blue_dmz'
    env.global_state.agent_energy[agent] = 50
    for other in ('blue_internal', 'blue_restricted'):
        env.event_queue.append(
            {
                'completion_tick': 10,
                'agent': other,
                'action': MagicMockAction(),
                'effect': ActionEffect(
                    success=True, state_deltas={}, observation_data={}
                ),
                'target_ip': None,
            }
        )
    initial_energy = env.global_state.agent_energy[agent]
    env.step({agent: 0})
    assert env.global_state.agent_energy[agent] < initial_energy


@pytest.mark.fast
def test_soc_per_agent_cap_blocks_third_self_action(env):
    env.reset(seed=42)
    agent = 'blue_dmz'
    env.global_state.agent_energy[agent] = 50
    for _ in range(2):
        env.event_queue.append(
            {
                'completion_tick': 999,
                'agent': agent,
                'action': MagicMockAction(),
                'effect': ActionEffect(
                    success=True, state_deltas={}, observation_data={}
                ),
                'target_ip': None,
            }
        )
    initial_energy = env.global_state.agent_energy[agent]
    env.step({agent: 0})
    assert env.global_state.agent_energy[agent] == initial_energy


@pytest.mark.fast
def test_agent_energy_exhaustion(env):
    env.reset(seed=42)
    agent = 'red_operator'
    env.global_state.agent_energy[agent] = 2
    env.step({agent: 0})
    assert len([e for e in env.event_queue if e['agent'] == agent]) == 0


@pytest.mark.fast
def test_dhcp_reallocation(env):
    env.reset(seed=42)
    initial_ips = set(env.global_state.all_hosts.keys())
    for _ in range(40):
        env.step({})
    new_ips = set(env.global_state.all_hosts.keys())
    assert initial_ips != new_ips


@pytest.mark.fast
def test_honeytoken_trap_alert(env):
    env.reset(seed=42)
    agent = 'red_operator'
    target_ip = next(
        (
            ip
            for ip, h in env.global_state.all_hosts.items()
            if h.subnet_cidr == '192.168.1.0/24'
        )
    )
    host = env.global_state.all_hosts[target_ip]
    host.contains_honeytokens = True
    host.vulnerabilities = ['MS17-010']
    env.global_state.agent_knowledge[agent] = {target_ip}
    env.global_state.action_history[agent] = {f'DiscoverNetworkServices:{target_ip}'}
    expected_effect = ActionEffect(
        success=True,
        state_deltas={f'hosts/{target_ip}/privilege': 'User'},
        observation_data={'exploit': target_ip},
    )
    with patch.object(ExploitEternalBlue, 'execute', return_value=expected_effect):
        action = ExploitEternalBlue(agent, target_ip)
        env.event_queue.append(
            {
                'agent': agent,
                'action': action,
                'completion_tick': 6,
                'effect': expected_effect,
                'target_ip': target_ip,
            }
        )
        for _ in range(6):
            env.step({})
        all_logs = env.global_state.siem_log_buffer
        honey_alerts = [
            log[0]
            for log in all_logs
            if isinstance(log[0], dict)
            and log[0].get('signature') == 'HONEYTOKEN_TRIGGERED'
        ]
        assert len(honey_alerts) > 0, (
            f'HONEYTOKEN_TRIGGERED missing. Buffer content: {all_logs}'
        )
        assert honey_alerts[0]['severity'] == 10
        assert honey_alerts[0]['target'] == target_ip
