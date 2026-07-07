import pytest

import netforge_rl.actions  # noqa: F401 — registers red actions
from netforge_rl.actions.attack_map import (
    ALL_TECHNIQUE_IDS,
    ATTACK_TECHNIQUES,
    technique_for,
)
from netforge_rl.baselines.policies import KillChainRedPolicy
from netforge_rl.core.registry import action_registry
from netforge_rl.environment.parallel_env import NetForgeRLEnv


@pytest.mark.fast
def test_every_red_action_class_is_mapped():
    """Each registered red action should have an ATT&CK technique (coverage taxonomy)."""
    for cls in action_registry._actions['red'].values():
        # Coordination (ShareIntelligence) is not an ATT&CK technique; everything else is.
        if cls.__name__ == 'ShareIntelligence':
            continue
        assert technique_for(cls.__name__) is not None, cls.__name__


@pytest.mark.fast
def test_technique_ids_are_well_formed():
    assert len(ALL_TECHNIQUE_IDS) >= 10
    for tid, name, tactic in ATTACK_TECHNIQUES.values():
        assert tid.startswith('T')
        assert name and tactic


@pytest.mark.integration
def test_attack_coverage_reported_in_info():
    env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 120})
    env.reset(seed=0)
    red = KillChainRedPolicy(seed=0)
    last = None
    import numpy as np

    while env.agents:
        _, _, term, trunc, last = env.step(
            {
                a: (red.act(env, a) if 'red' in a else np.array([0, 0]))
                for a in env.agents
            }
        )
        if all(term.values()) or all(trunc.values()):
            break
    info = next(v for k, v in last.items() if 'red' in k)
    assert info['attack_coverage'] > 0.0
    assert (
        'T1046' in info['attack_techniques']
    )  # discovery — the kill-chain always recons
    assert set(info['attack_techniques']) <= ALL_TECHNIQUE_IDS
