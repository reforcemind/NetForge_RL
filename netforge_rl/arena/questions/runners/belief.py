from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from netforge_rl.arena.evaluate import run_episode
from netforge_rl.arena.questions.policies import blue, killchain


def run_belief_vs_oracle(*, seeds: Sequence[int], max_ticks: int) -> dict:
    from netforge_rl.core.graph_obs import build_graph_observation, build_oracle_graph
    from netforge_rl.environment.graph_wrapper import GraphObservationWrapper
    from netforge_rl.environment.parallel_env import NetForgeRLEnv

    seed = int(seeds[0])
    env = GraphObservationWrapper(
        NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': max_ticks}),
        oracle=True,
    )
    env.reset(seed=seed)
    live = [
        ip
        for ip, h in env.unwrapped.global_state.all_hosts.items()
        if not ip.startswith('169.254.')
    ]
    planted = min(live)
    host = env.unwrapped.global_state.all_hosts[planted]
    host.compromised_by = 'red_operator'
    host.privilege = 'User'

    gs = env.unwrapped.global_state
    belief = build_graph_observation(gs, agent_id='blue_dmz')
    oracle = build_oracle_graph(gs)
    vis_b = int(belief['node_mask'].sum())
    vis_o = max(int(oracle['node_mask'].sum()), 1)
    oracle_comp = set(np.where(oracle['node_features'][:, 2] > 0.5)[0].tolist())
    belief_comp = set(np.where(belief['node_features'][:, 0] > 0.5)[0].tolist())
    recall = len(oracle_comp & belief_comp) / max(len(oracle_comp), 1)

    rec = run_episode(
        killchain(),
        blue(),
        scenario='ransomware',
        seed=seed,
        max_ticks=max_ticks,
        graph_obs=True,
    )
    return {
        'planted_host': planted,
        'visible_belief': vis_b,
        'visible_oracle': vis_o,
        'visible_ratio': round(vis_b / vis_o, 4),
        'oracle_compromised_nodes': len(oracle_comp),
        'belief_flagged_nodes': len(belief_comp),
        'compromise_recall': round(recall, 4),
        'heuristic_episode': rec.metrics.as_dict(),
    }
