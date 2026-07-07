from __future__ import annotations

from typing import Optional

from netforge_rl.environment.parallel_env import NetForgeRLEnv

# Held-out seeds for the generalization split; used with evaluation_mode=True.
EVAL_SEEDS: tuple[int, ...] = (
    9001,
    9002,
    9003,
    9004,
    9005,
    9006,
    9007,
    9008,
    9009,
    9010,
    9011,
    9012,
    9013,
    9014,
    9015,
    9016,
    9017,
    9018,
    9019,
    9020,
)

DIFFICULTY_PRESETS: dict[str, dict] = {
    'easy': {
        'max_active_hosts': 6,
        'dhcp_interval': 0,
        'topology_churn_rate': 0.0,
        'topology_migration_rate': 0.0,
        'topology_arrival_rate': 0.0,
        'log_latency': 0,
        'max_ticks': 200,
    },
    'medium': {
        'max_active_hosts': 15,
        'dhcp_interval': 80,
        'topology_churn_rate': 0.01,
        'topology_migration_rate': 0.0,
        'topology_arrival_rate': 0.0,
        'log_latency': 2,
        'max_ticks': 200,
    },
    'hard': {
        'max_active_hosts': 100,
        'dhcp_interval': 40,
        'topology_churn_rate': 0.02,
        'topology_migration_rate': 0.01,
        'topology_arrival_rate': 0.005,
        'log_latency': 4,
        'max_ticks': 200,
    },
}


def make_config(
    difficulty: str = 'medium',
    scenario_type: str = 'ransomware',
    evaluation: bool = False,
    **overrides,
) -> dict:
    """Env config for a difficulty tier; overrides win over the preset."""
    if difficulty not in DIFFICULTY_PRESETS:
        raise KeyError(
            f'Unknown difficulty {difficulty!r}. '
            f'Choose from {sorted(DIFFICULTY_PRESETS)}.'
        )
    cfg = dict(DIFFICULTY_PRESETS[difficulty])
    cfg['scenario_type'] = scenario_type
    cfg['evaluation_mode'] = evaluation
    cfg.update(overrides)
    return cfg


def make_env(
    difficulty: str = 'medium',
    scenario_type: str = 'ransomware',
    evaluation: bool = False,
    seed: Optional[int] = None,
    **overrides,
) -> NetForgeRLEnv:
    """Build (and reset, if seed is given) an env at a difficulty tier."""
    env = NetForgeRLEnv(make_config(difficulty, scenario_type, evaluation, **overrides))
    if seed is not None:
        env.reset(seed=seed)
    return env
