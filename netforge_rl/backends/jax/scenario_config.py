SCENARIO_RANSOMWARE = 0
SCENARIO_APT = 1
SCENARIO_CLOUD = 2
SCENARIO_IOT = 3
SCENARIO_OT = 4

SCENARIO_IDS = {
    'ransomware': SCENARIO_RANSOMWARE,
    'apt_espionage': SCENARIO_APT,
    'cloud_hybrid': SCENARIO_CLOUD,
    'iot_grid': SCENARIO_IOT,
    'ot_stuxnet': SCENARIO_OT,
}

_RED_WEIGHT_KEYS = (
    'user-compromise',
    'root-privesc',
    'host-impact',
    'kinetic',
    'exfil/host',
    'dc-compromise',
    'recon',
)
_BLUE_WEIGHT_KEYS = (
    'good-isolate',
    'bad-isolate',
    'restore',
    'health-ratio',
    'dc-loss',
    'deception',
)

# scenario -> (red weights matching _RED_WEIGHT_KEYS, blue weights matching _BLUE_WEIGHT_KEYS)
_RW = {
    SCENARIO_RANSOMWARE: (
        (1.0, 3.0, 10.0, 30.0, 1.0, 2.0, 0.2),
        (5.0, 2.0, 3.0, 2.0, 5.0, 0.5),
    ),
    SCENARIO_APT: (
        (2.0, 4.0, 20.0, 25.0, 3.0, 5.0, 1.0),
        (20.0, 1.0, 3.0, 1.0, 8.0, 0.4),
    ),
    SCENARIO_CLOUD: (
        (1.5, 3.0, 12.0, 25.0, 2.0, 25.0, 0.3),
        (5.0, 1.0, 3.0, 1.5, 30.0, 0.4),
    ),
    SCENARIO_IOT: (
        (2.0, 4.0, 10.0, 20.0, 1.0, 40.0, 0.3),
        (3.0, 1.0, 3.0, 0.5, 40.0, 0.4),
    ),
    SCENARIO_OT: (
        (4.0, 8.0, 12.0, 60.0, 2.0, 12.0, 0.3),
        (6.0, 1.0, 3.0, 0.5, 12.0, 0.4),
    ),
}

_RED_SCALE = 100.0
_BLUE_SCALE = 10.0
