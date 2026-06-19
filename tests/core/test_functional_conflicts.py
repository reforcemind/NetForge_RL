import copy
import pytest
from netforge_rl.core.action import ActionEffect
from netforge_rl.core.functional import resolve_conflicts
from netforge_rl.core.physics import ConflictResolutionEngine


def _eff(success: bool, deltas, obs=None) -> ActionEffect:
    return ActionEffect(
        success=success, state_deltas=deltas, observation_data=obs or {}
    )


@pytest.mark.fast
def test_no_collision_passes_red_unchanged() -> None:
    effects = {
        'red_operator': _eff(True, {'hosts/1.1.1.1/privilege': 'User'}),
        'blue_dmz': _eff(True, {'hosts/2.2.2.2/status': 'isolated'}),
    }
    out = resolve_conflicts(effects)
    legacy = ConflictResolutionEngine.resolve(copy.deepcopy(effects))
    assert out['red_operator'].success is True
    assert legacy['red_operator'].success is True
    assert out['blue_dmz'].success is True


@pytest.mark.fast
def test_collision_nullifies_red() -> None:
    effects = {
        'red_operator': _eff(True, {'hosts/1.1.1.1/privilege': 'Root'}),
        'blue_dmz': _eff(True, {'hosts/1.1.1.1/status': 'isolated'}),
    }
    out = resolve_conflicts(effects)
    legacy = ConflictResolutionEngine.resolve(copy.deepcopy(effects))
    assert out['red_operator'].success is False
    assert legacy['red_operator'].success is False
    assert (
        out['red_operator'].observation_data['alert']
        == 'TEMPORAL_COLLISION_DEFENSE_SUPREMACY'
    )
    assert out['blue_dmz'].success is True


@pytest.mark.fast
def test_collision_resets_dict_deltas_to_empty_dict() -> None:
    effects = {
        'red_operator': _eff(True, {'hosts/1.1.1.1/privilege': 'Root'}),
        'blue_dmz': _eff(True, {'hosts/1.1.1.1/status': 'isolated'}),
    }
    out = resolve_conflicts(effects)
    assert out['red_operator'].state_deltas == {}


@pytest.mark.fast
def test_collision_resets_list_deltas_to_empty_list() -> None:

    class _Cmd:
        def __init__(self, ip):
            self.target_ip = ip

    effects = {
        'red_operator': _eff(True, [_Cmd('1.1.1.1')]),
        'blue_dmz': _eff(True, {'hosts/1.1.1.1/status': 'isolated'}),
    }
    out = resolve_conflicts(effects)
    assert out['red_operator'].success is False
    assert out['red_operator'].state_deltas == []


@pytest.mark.fast
def test_failed_blue_does_not_defend() -> None:
    effects = {
        'red_operator': _eff(True, {'hosts/1.1.1.1/privilege': 'Root'}),
        'blue_dmz': _eff(False, {'hosts/1.1.1.1/status': 'isolated'}),
    }
    out = resolve_conflicts(effects)
    assert out['red_operator'].success is True


@pytest.mark.fast
def test_already_failed_red_passes_through() -> None:
    effects = {
        'red_operator': _eff(False, {'hosts/1.1.1.1/privilege': 'Root'}),
        'blue_dmz': _eff(True, {'hosts/1.1.1.1/status': 'isolated'}),
    }
    out = resolve_conflicts(effects)
    assert out['red_operator'].success is False
    assert 'alert' not in out['red_operator'].observation_data


@pytest.mark.fast
def test_none_effect_passes_through() -> None:
    effects = {
        'red_operator': None,
        'blue_dmz': _eff(True, {'hosts/1.1.1.1/status': 'isolated'}),
    }
    out = resolve_conflicts(effects)
    assert out['red_operator'] is None


@pytest.mark.fast
def test_input_dict_is_not_mutated() -> None:
    red_eff = _eff(True, {'hosts/1.1.1.1/privilege': 'Root'})
    effects = {
        'red_operator': red_eff,
        'blue_dmz': _eff(True, {'hosts/1.1.1.1/status': 'isolated'}),
    }
    resolve_conflicts(effects)
    assert red_eff.success is True
    assert red_eff.state_deltas == {'hosts/1.1.1.1/privilege': 'Root'}
    assert 'alert' not in red_eff.observation_data


@pytest.mark.fast
def test_returns_new_dict_instance() -> None:
    effects = {'blue_dmz': _eff(True, {'hosts/1.1.1.1/status': 'isolated'})}
    out = resolve_conflicts(effects)
    assert out is not effects
