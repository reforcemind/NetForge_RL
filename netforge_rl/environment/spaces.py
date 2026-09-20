import gymnasium as gym
import numpy as np

from netforge_rl.environment.constants import (
    ACTION_MASK_DIM,
    ADJ_FLAT_DIM,
    BLUE_COMM_DIM,
    N_ACTION_TYPES,
    N_HOST_SLOTS,
    OBS_VECTOR_DIM,
)
from netforge_rl.nlp.log_encoder import EMBEDDING_DIM
from netforge_rl.siem.pcap_synthesizer import N_PACKETS, NODE_DIM, PACKET_DIM

FLAT_ACTION_MASK = ACTION_MASK_DIM


def _split_flat_mask(mask, nvec):
    """PettingZoo passes a concatenated action_mask; Gymnasium wants a tuple."""
    if mask is None or isinstance(mask, tuple):
        return mask
    arr = np.asarray(mask)
    expected = int(np.sum(nvec))
    if arr.ndim == 1 and arr.size == expected:
        cuts = np.cumsum(np.asarray(nvec, dtype=int))[:-1]
        parts = np.split(arr.astype(np.int8, copy=False), cuts)
        return tuple(parts)
    return mask


class CyberActionSpace(gym.spaces.MultiDiscrete):
    """``MultiDiscrete([types, hosts])`` that accepts a flat PettingZoo mask."""

    def sample(self, mask=None, probability=None):
        return super().sample(
            mask=_split_flat_mask(mask, self.nvec),
            probability=_split_flat_mask(probability, self.nvec),
        )


def build_observation_spaces(possible_agents, pcap_obs: bool) -> dict:
    """Per-agent Dict observation spaces. Blue agents also get a shared comm channel."""
    base = {
        'obs': gym.spaces.Box(-1.0, 1.0, shape=(OBS_VECTOR_DIM,), dtype=np.float32),
        'action_mask': gym.spaces.Box(0, 1, shape=(ACTION_MASK_DIM,), dtype=np.int8),
        'siem_embedding': gym.spaces.Box(
            -1.0, 1.0, shape=(EMBEDDING_DIM,), dtype=np.float32
        ),
        'adj_matrix': gym.spaces.Box(0.0, 1.0, shape=(ADJ_FLAT_DIM,), dtype=np.float32),
        'delta_t': gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
    }
    if pcap_obs:
        base['pcap'] = gym.spaces.Box(
            0.0, 1.0, shape=(N_PACKETS, PACKET_DIM), dtype=np.float32
        )
        base['node_features'] = gym.spaces.Box(
            0.0, 1.0, shape=(N_HOST_SLOTS, NODE_DIM), dtype=np.float32
        )
    blue = dict(base)
    blue['blue_comm'] = gym.spaces.Box(
        0.0, 1.0, shape=(BLUE_COMM_DIM,), dtype=np.float32
    )
    return {
        agent: gym.spaces.Dict(blue if 'blue' in agent else base)
        for agent in possible_agents
    }


def build_action_spaces(possible_agents) -> dict:
    """MultiDiscrete([action_type, target_host_index]) per agent."""
    return {
        agent: CyberActionSpace([N_ACTION_TYPES, N_HOST_SLOTS])
        for agent in possible_agents
    }
