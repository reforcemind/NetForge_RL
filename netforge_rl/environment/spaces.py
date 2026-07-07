import gymnasium as gym
import numpy as np

from netforge_rl.nlp.log_encoder import EMBEDDING_DIM
from netforge_rl.siem.pcap_synthesizer import N_PACKETS, NODE_DIM, PACKET_DIM


def build_observation_spaces(possible_agents, pcap_obs: bool) -> dict:
    """Per-agent Dict observation spaces. Blue agents also get a shared comm channel."""
    base = {
        'obs': gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=np.float32),
        'action_mask': gym.spaces.Box(0, 1, shape=(32 + 100,), dtype=np.int8),
        'siem_embedding': gym.spaces.Box(
            -1.0, 1.0, shape=(EMBEDDING_DIM,), dtype=np.float32
        ),
        'adj_matrix': gym.spaces.Box(0.0, 1.0, shape=(10000,), dtype=np.float32),
        'delta_t': gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
    }
    if pcap_obs:
        base['pcap'] = gym.spaces.Box(
            0.0, 1.0, shape=(N_PACKETS, PACKET_DIM), dtype=np.float32
        )
        base['node_features'] = gym.spaces.Box(
            0.0, 1.0, shape=(100, NODE_DIM), dtype=np.float32
        )
    blue = dict(base)
    blue['blue_comm'] = gym.spaces.Box(0.0, 1.0, shape=(100,), dtype=np.float32)
    return {
        agent: gym.spaces.Dict(blue if 'blue' in agent else base)
        for agent in possible_agents
    }


def build_action_spaces(possible_agents) -> dict:
    """MultiDiscrete([action_type, target_host_index]) per agent."""
    return {agent: gym.spaces.MultiDiscrete([32, 100]) for agent in possible_agents}
