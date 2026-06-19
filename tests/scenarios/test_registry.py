import pytest
from netforge_rl.scenarios import _SCENARIOS, get_scenario_class
from netforge_rl.scenarios.apt_espionage import AptEspionageScenario
from netforge_rl.scenarios.iot_grid import IoTGridScenario
from netforge_rl.scenarios.ot_stuxnet import OTStuxnetScenario
from netforge_rl.scenarios.ransomware import RansomwareScenario
from netforge_rl.scenarios.cloud_hybrid import CloudHybridScenario


@pytest.mark.fast
def test_builtins_available() -> None:
    assert set(_SCENARIOS.keys()) >= {
        'ransomware',
        'apt_espionage',
        'iot_grid',
        'ot_stuxnet',
        'cloud_hybrid',
    }


@pytest.mark.fast
@pytest.mark.parametrize(
    'name,cls',
    [
        ('ransomware', RansomwareScenario),
        ('apt_espionage', AptEspionageScenario),
        ('iot_grid', IoTGridScenario),
        ('ot_stuxnet', OTStuxnetScenario),
        ('cloud_hybrid', CloudHybridScenario),
    ],
)
def test_lookup(name, cls) -> None:
    assert get_scenario_class(name) is cls


@pytest.mark.fast
def test_unknown_raises_with_known_names() -> None:
    with pytest.raises(KeyError, match='Available'):
        get_scenario_class('does_not_exist')
