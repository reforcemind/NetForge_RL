from netforge_rl.scenarios.ransomware import RansomwareScenario
from netforge_rl.scenarios.apt_espionage import AptEspionageScenario
from netforge_rl.scenarios.iot_grid import IoTGridScenario
from netforge_rl.scenarios.ot_stuxnet import OTStuxnetScenario
from netforge_rl.scenarios.cloud_hybrid import CloudHybridScenario

_SCENARIOS = {
    'ransomware': RansomwareScenario,
    'apt_espionage': AptEspionageScenario,
    'iot_grid': IoTGridScenario,
    'ot_stuxnet': OTStuxnetScenario,
    'cloud_hybrid': CloudHybridScenario,
}


def get_scenario_class(name: str):
    key = name.lower()
    if key not in _SCENARIOS:
        raise KeyError(f'Unknown scenario {name!r}. Available: {sorted(_SCENARIOS)}')
    return _SCENARIOS[key]
