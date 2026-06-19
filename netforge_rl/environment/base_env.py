import abc
import numpy as np
from pettingzoo import ParallelEnv
from typing import Dict, Tuple, Any


class BaseNetForgeRLEnv(ParallelEnv, abc.ABC):
    """Abstract Base Class for all Continuous-Time MARL environments."""

    @abc.abstractmethod
    def __init__(self, scenario_config: dict):
        """Initialize scenario and network state."""
        pass

    @abc.abstractmethod
    def reset(
        self, seed=None, options=None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """Reset environment and return initial observations."""
        pass

    @abc.abstractmethod
    def step(
        self, agent_actions: Dict[str, Any]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """Core physics loop. Returns (obs, rewards, term, trunc, infos)."""
        pass
