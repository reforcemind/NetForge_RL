import pytest

from netforge_rl.baselines.policies import KillChainRedPolicy
from netforge_rl.environment.parallel_env import NetForgeRLEnv


def _run(env, red):
    last = None
    while env.agents:
        _, _, term, trunc, last = env.step({a: red.act(env, a) for a in env.agents})
        if all(term.values()) or all(trunc.values()):
            break
    return next(v for k, v in last.items() if 'red' in k)


@pytest.mark.integration
def test_deception_hits_counted_when_red_strikes_decoys():
    env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 120})
    env.reset(seed=0)
    for ip, h in env.global_state.all_hosts.items():
        if not ip.startswith('169.254.'):
            h.decoy = 'active'
    info = _run(env, KillChainRedPolicy(seed=0))
    assert info['deception_hits'] > 0
    assert 0.0 <= info['deception_efficacy'] <= 1.0


@pytest.mark.integration
def test_no_deception_hits_without_decoys():
    env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 120})
    env.reset(seed=0)
    for h in env.global_state.all_hosts.values():
        h.decoy = 'inactive'
        h.contains_honeytokens = False
    info = _run(env, KillChainRedPolicy(seed=0))
    assert info['deception_hits'] == 0.0
    assert info['deception_efficacy'] == 0.0
