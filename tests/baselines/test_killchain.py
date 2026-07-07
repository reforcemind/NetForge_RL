import pytest

from netforge_rl.baselines.policies import (
    HeuristicRedPolicy,
    KillChainRedPolicy,
    RandomPolicy,
)
from netforge_rl.environment.parallel_env import NetForgeRLEnv, PADDING_SUBNET


def _run(red_policy, scenario='ransomware', seed=0, max_ticks=150):
    env = NetForgeRLEnv({'scenario_type': scenario, 'max_ticks': max_ticks})
    env.reset(seed=seed)
    blue = RandomPolicy(seed=seed)
    while env.agents:
        actions = {
            a: (red_policy.act(env, a) if 'red' in a else blue.act(env, a))
            for a in env.agents
        }
        _, _, term, trunc, _ = env.step(actions)
        if all(term.values()) or all(trunc.values()):
            break
    active = [
        h
        for h in env.global_state.all_hosts.values()
        if h.subnet_cidr != PADDING_SUBNET
    ]
    return sum(1 for h in active if h.compromised_by != 'None')


@pytest.mark.integration
def test_killchain_red_actually_compromises_hosts():
    """The kill-chain baseline must land exploits; the naive heuristic never does
    (it skips recon, so ExploitRemoteService fails its prior-state check)."""
    kc = _run(KillChainRedPolicy(seed=0))
    naive = _run(HeuristicRedPolicy(seed=0))
    assert kc >= 1, 'kill-chain red compromised nothing — benchmark not stressed'
    assert kc > naive


@pytest.mark.integration
def test_killchain_red_is_seed_deterministic():
    assert _run(KillChainRedPolicy(seed=3)) == _run(KillChainRedPolicy(seed=3))
