import pytest
from netforge_rl.scenarios import available, get_scenario_class, register
from netforge_rl.scenarios.apt_espionage import AptEspionageScenario
from netforge_rl.scenarios.base_scenario import BaseScenario
from netforge_rl.scenarios.iot_grid import IoTGridScenario
from netforge_rl.scenarios.ot_stuxnet import OTStuxnetScenario
from netforge_rl.scenarios.ransomware import RansomwareScenario

@pytest.mark.fast
def test_builtins_available() -> None:
    assert set(available()) >= {'ransomware', 'apt_espionage', 'iot_grid', 'ot_stuxnet'}

@pytest.mark.fast
@pytest.mark.parametrize('name,cls', [('ransomware', RansomwareScenario), ('apt_espionage', AptEspionageScenario), ('iot_grid', IoTGridScenario), ('ot_stuxnet', OTStuxnetScenario), ('OT_STUXNET', OTStuxnetScenario)])
def test_lookup(name, cls) -> None:
    assert get_scenario_class(name) is cls

@pytest.mark.fast
def test_unknown_raises_with_known_names() -> None:
    with pytest.raises(KeyError, match='Available'):
        get_scenario_class('does_not_exist')

@pytest.mark.fast
def test_register_custom_scenario() -> None:

    class Custom(BaseScenario):

        def __init__(self, agents):
            self.agents = agents

        def calculate_reward(self, *a, **k):
            return 0.0

        def check_termination(self, state):
            return {a: False for a in self.agents}
    register('custom_test', lambda: Custom)
    assert get_scenario_class('custom_test') is Custom