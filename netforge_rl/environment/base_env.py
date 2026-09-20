import abc

import numpy as np
from pettingzoo import ParallelEnv


class BaseNetForgeRLEnv(ParallelEnv, abc.ABC):
    """Abstract PettingZoo parallel env."""

    @abc.abstractmethod
    def __init__(self, scenario_config: dict):
        pass

    @abc.abstractmethod
    def reset(
        self, seed=None, options=None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        pass

    @abc.abstractmethod
    def step(
        self, agent_actions: dict[str, object]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        pass
