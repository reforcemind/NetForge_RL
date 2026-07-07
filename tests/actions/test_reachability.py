import pytest

import netforge_rl.actions  # noqa: F401 — registers every action
from netforge_rl.core.registry import action_registry, team_of
from netforge_rl.environment.parallel_env import NetForgeRLEnv

DEFAULT_AGENTS = ['red_operator', 'blue_dmz', 'blue_internal', 'blue_restricted']


@pytest.mark.fast
@pytest.mark.parametrize('agent_id', DEFAULT_AGENTS)
def test_every_registered_action_is_reachable(agent_id):
    """Each action registered for a team must resolve for that team's live agents."""
    team = team_of(agent_id)
    for group_id, cls in action_registry._actions[team].items():
        resolved = action_registry.get_action_class(agent_id, group_id)
        assert resolved is cls, f'{cls.__name__} unreachable for {agent_id}'


@pytest.mark.fast
def test_action_ids_fit_in_action_space():
    """Every action id must be addressable by the MultiDiscrete([32, 100]) space."""
    for team in ('red', 'blue'):
        for group_id in action_registry._actions[team]:
            assert group_id < 32


@pytest.mark.fast
def test_blue_detection_actions_are_reachable():
    """Regression: Monitor/Analyze/DeployEDR/DeployDecoy were unreachable before
    the taxonomy was unified under a single team per role."""
    env = NetForgeRLEnv({'scenario_type': 'ransomware'})
    reachable = {
        action_registry.get_action_class('blue_dmz', gid).__name__
        for gid in action_registry._actions['blue']
    }
    assert {'Monitor', 'Analyze', 'DeployEDR', 'DeployDecoy'} <= reachable
    mask = env.action_mask('blue_dmz')
    for name in ('Monitor', 'Analyze', 'DeployEDR', 'DeployDecoy'):
        gid = next(
            g for g, c in action_registry._actions['blue'].items() if c.__name__ == name
        )
        assert mask[gid] == 1
