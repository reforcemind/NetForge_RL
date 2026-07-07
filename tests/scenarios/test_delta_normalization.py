import pytest

from netforge_rl.core.action import ActionEffect
from netforge_rl.core.commands import (
    UpdateHostPrivilegeCommand,
    UpdateHostStatusCommand,
    iter_host_deltas,
)
from netforge_rl.scenarios.ransomware import RansomwareScenario

AGENTS = ['red_operator', 'blue_dmz']


@pytest.mark.fast
def test_iter_host_deltas_handles_both_encodings():
    dict_form = {'hosts/10.0.0.5/privilege': 'User', 'knowledge/x/y': 'True'}
    list_form = [UpdateHostPrivilegeCommand('10.0.0.5', 'User', compromised_by='red')]
    assert ('privilege', '10.0.0.5', 'User') in list(iter_host_deltas(dict_form))
    got = list(iter_host_deltas(list_form))
    assert ('privilege', '10.0.0.5', 'User') in got
    assert ('compromised_by', '10.0.0.5', 'red') in got


@pytest.mark.fast
def test_command_and_dict_privilege_reward_match(global_state):
    """A privilege gain must be rewarded identically whether the action returns a
    dict delta or a command-list delta (the P0 benchmark-validity fix)."""
    scenario = RansomwareScenario(AGENTS)
    ip = next(iter(global_state.all_hosts))

    # Both encodings describe the same effect: privilege -> User, compromised_by -> red.
    # (Dict exploits like ExploitBlueKeep set both keys; the command carries both.)
    dict_effect = ActionEffect(
        success=True,
        state_deltas={
            f'hosts/{ip}/privilege': 'User',
            f'hosts/{ip}/compromised_by': 'red',
        },
        observation_data={},
    )
    list_effect = ActionEffect(
        success=True,
        state_deltas=[UpdateHostPrivilegeCommand(ip, 'User', compromised_by='red')],
        observation_data={},
    )
    r_dict = scenario.calculate_reward('red_operator', global_state, dict_effect)
    r_list = scenario.calculate_reward('red_operator', global_state, list_effect)
    assert r_dict == r_list
    assert r_list > 0


@pytest.mark.fast
def test_isolation_command_credits_blue(global_state):
    scenario = RansomwareScenario(AGENTS)
    ip = next(iter(global_state.all_hosts))
    global_state.all_hosts[ip].compromised_by = 'red_operator'
    effect = ActionEffect(
        success=True,
        state_deltas=[UpdateHostStatusCommand(ip, 'isolated')],
        observation_data={},
    )
    reward = scenario.calculate_reward('blue_dmz', global_state, effect)
    assert reward > 0
