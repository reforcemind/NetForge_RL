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


def is_builtin(name: str) -> bool:
    return str(name).lower() in _SCENARIOS


def get_scenario_class(name: str):
    key = name.lower()
    if key in _SCENARIOS:
        return _SCENARIOS[key]
    from netforge_rl.scenarios.yaml_dsl import PACKS_DIR, load_scenario_file

    pack = PACKS_DIR / f'{key}.yaml'
    if pack.exists():
        spec = load_scenario_file(pack)
        base = spec.base.lower()
        if base in _SCENARIOS:
            return _SCENARIOS[base]
    raise KeyError(f'Unknown scenario {name!r}. Available: {sorted(_SCENARIOS)}')


def get_reward_weights(name: str) -> dict:
    """Return the documented reward decomposition for a scenario."""
    return get_scenario_class(name).REWARD_WEIGHTS
