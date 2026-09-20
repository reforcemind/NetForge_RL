from __future__ import annotations

import numpy as np
import pytest

from netforge_rl.core.graph_obs import build_graph_observation, build_oracle_graph
from netforge_rl.environment.graph_wrapper import GraphObservationWrapper
from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.siem.pcap_synthesizer import NODE_DIM


@pytest.fixture
def env():
    e = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 20})
    e.reset(seed=0)
    return e


@pytest.mark.fast
def test_graph_shapes_and_types(env):
    g = build_graph_observation(env.global_state)
    assert g['node_features'].shape == (100, NODE_DIM)
    assert g['edge_index'].shape[0] == 2
    assert g['edge_index'].dtype == np.int64
    assert g['edge_attr'].shape == (g['n_edges'], 1)
    assert g['edge_index'].shape[1] == g['n_edges']


@pytest.mark.fast
def test_padding_hosts_are_masked(env):
    g = build_graph_observation(env.global_state)
    assert int(g['node_mask'].sum()) == len(env._active_hosts())


@pytest.mark.fast
def test_red_fog_of_war_hides_undiscovered_hosts(env):
    oracle = build_oracle_graph(env.global_state)
    red = build_graph_observation(env.global_state, agent_id='red_operator')
    assert red['node_mask'].sum() < oracle['node_mask'].sum()


@pytest.mark.fast
def test_blue_belief_does_not_leak_compromise(env):
    live = [
        ip
        for ip, h in env.global_state.all_hosts.items()
        if not ip.startswith('169.254.')
    ]
    ip = sorted(live)[0]
    env.global_state.all_hosts[ip].compromised_by = 'red_operator'
    env.global_state.all_hosts[ip].privilege = 'Root'
    idx = sorted(env.global_state.all_hosts.keys()).index(ip)
    belief = build_graph_observation(
        env.global_state, agent_id='blue_dmz', mode='belief'
    )
    oracle = build_graph_observation(env.global_state, mode='oracle')
    assert oracle['node_features'][idx, 2] == 1.0
    assert belief['node_features'][idx, 0] == 0.0


@pytest.mark.fast
def test_edges_only_between_visible_nodes(env):
    g = build_graph_observation(env.global_state, agent_id='blue_dmz')
    visible = set(np.where(g['node_mask'] > 0)[0].tolist())
    for s, d in g['edge_index'].T.tolist():
        assert s in visible and d in visible


@pytest.mark.integration
def test_wrapper_injects_belief_and_optional_oracle():
    env = GraphObservationWrapper(NetForgeRLEnv({'scenario_type': 'ransomware'}))
    obs, infos = env.reset(seed=0)
    assert 'graph' in infos['blue_dmz']
    assert 'oracle_graph' not in infos['blue_dmz']
    assert infos['blue_dmz']['graph']['node_features'].shape == (100, NODE_DIM)

    env = GraphObservationWrapper(
        NetForgeRLEnv({'scenario_type': 'ransomware'}), oracle=True
    )
    _, infos = env.reset(seed=0)
    assert 'oracle_graph' in infos['blue_dmz']
    assert int(infos['blue_dmz']['oracle_graph']['node_mask'].sum()) >= int(
        infos['blue_dmz']['graph']['node_mask'].sum()
    )
