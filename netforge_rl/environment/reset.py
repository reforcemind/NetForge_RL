from __future__ import annotations

import numpy as np

from netforge_rl.core.observation import BaseObservation
from netforge_rl.environment.constants import (
    BLUE_COMM_DIM,
    N_HOST_SLOTS,
    OBS_VECTOR_DIM,
)
from netforge_rl.nlp.log_encoder import EMBEDDING_DIM
from netforge_rl.siem.pcap_synthesizer import N_PACKETS, NODE_DIM, PACKET_DIM


def empty_episode_metrics() -> dict:
    return {
        'infection_times': {},
        'detection_times': {},
        'isolation_times': {},
        'exfiltrated_data': 0.0,
        'sla_uptime_sum': 0.0,
        'steps_count': 0,
        'deception_hits': 0,
        'red_actions': 0,
        'attack_techniques': set(),
        'false_positives': 0,
    }


def seed_agent_budgets(env) -> None:
    budgets = env.config.budgets
    env.global_state.agent_energy = {agent: budgets.energy for agent in env.agents}
    env.global_state.agent_funds = {
        agent: budgets.blue_funds if 'blue' in agent else budgets.red_funds
        for agent in env.agents
    }
    env.global_state.agent_compute = {agent: budgets.compute for agent in env.agents}
    env.global_state.business_downtime_score = 0.0
    env.global_state.siem_log_buffer = []


def initial_observations(env) -> dict[str, dict]:
    observations = {}
    for agent_id in env.agents:
        obs = BaseObservation(agent_id)
        obs.update_from_state(env.global_state, [])
        agent_obs = {
            'obs': obs.to_numpy(max_size=OBS_VECTOR_DIM),
            'action_mask': env._cached_action_masks[agent_id],
            'siem_embedding': np.zeros(EMBEDDING_DIM, dtype=np.float32),
            'adj_matrix': env.global_state.get_adjacency_matrix().flatten(),
            'delta_t': np.zeros(1, dtype=np.float32),
        }
        if 'blue' in agent_id:
            agent_obs['blue_comm'] = np.zeros(BLUE_COMM_DIM, dtype=np.float32)
        if env.pcap_obs:
            agent_obs['pcap'] = np.zeros((N_PACKETS, PACKET_DIM), dtype=np.float32)
            agent_obs['node_features'] = np.zeros(
                (N_HOST_SLOTS, NODE_DIM), dtype=np.float32
            )
        observations[agent_id] = agent_obs
    return observations
