import pytest
from netforge_rl.core.physics import ConflictResolutionEngine
from netforge_rl.core.action import ActionEffect


@pytest.mark.fast
def test_physics_conflict_resolution_blue_wins():
    cre = ConflictResolutionEngine()
    red_eff = ActionEffect(
        success=True,
        state_deltas={'hosts/10.0.0.5/privilege': 'Root'},
        observation_data={'exploit': '10.0.0.5'},
    )
    blue_eff = ActionEffect(
        success=True,
        state_deltas={'hosts/10.0.0.5/status': 'isolated'},
        observation_data={},
    )
    effects = {'red_operator_0': red_eff, 'blue_operator_0': blue_eff}
    resolved = cre.resolve(effects)
    assert resolved['red_operator_0'].success is False
    assert resolved['red_operator_0'].state_deltas == {}
    assert (
        resolved['red_operator_0'].observation_data['alert']
        == 'TEMPORAL_COLLISION_DEFENSE_SUPREMACY'
    )
    assert resolved['blue_operator_0'].success is True
    assert 'hosts/10.0.0.5/status' in resolved['blue_operator_0'].state_deltas


@pytest.mark.fast
def test_physics_no_conflict_different_nodes():
    cre = ConflictResolutionEngine()
    red_eff = ActionEffect(
        success=True,
        state_deltas={'hosts/10.0.0.5/privilege': 'Root'},
        observation_data={},
    )
    blue_eff = ActionEffect(
        success=True,
        state_deltas={'hosts/10.0.2.1/status': 'isolated'},
        observation_data={},
    )
    effects = {'red_operator_0': red_eff, 'blue_operator_0': blue_eff}
    resolved = cre.resolve(effects)
    assert resolved['red_operator_0'].success is True
    assert resolved['blue_operator_0'].success is True
