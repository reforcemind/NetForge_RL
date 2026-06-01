"""Reference policies. Each ``act`` returns an int64[2] action array."""

from __future__ import annotations

import random
from typing import Protocol

import numpy as np


class BasePolicy(Protocol):
    name: str

    def act(self, env, agent_id: str) -> np.ndarray: ...


class RandomPolicy:
    """Uniform random over the agent's MultiDiscrete([32, 100]) action space."""

    name = 'random'

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def act(self, env, agent_id: str) -> np.ndarray:
        space = env.action_space(agent_id)
        return np.array(
            [self._rng.integers(0, n) for n in space.nvec], dtype=np.int64
        )


class HeuristicBluePolicy:
    """Blue: isolate the first compromised host; otherwise scan/analyze.

    No-ops if no compromised host is visible; small but informative
    baseline.
    """

    name = 'heuristic-blue'

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def act(self, env, agent_id: str) -> np.ndarray:
        target_ips = sorted(env.global_state.all_hosts.keys())
        compromised = [
            ip for ip, h in env.global_state.all_hosts.items()
            if h.compromised_by != 'None' and h.status != 'isolated'
        ]
        if compromised:
            ip = self._rng.choice(compromised)
            return np.array(
                [0, target_ips.index(ip)],  # action 0 = IsolateHost for blue team
                dtype=np.int64,
            )
        return np.array([3, 0], dtype=np.int64)  # Analyze on host 0


class HeuristicRedPolicy:
    """Red: exploit the first uncompromised host with a known CVE."""

    name = 'heuristic-red'

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def act(self, env, agent_id: str) -> np.ndarray:
        target_ips = sorted(env.global_state.all_hosts.keys())
        targets = [
            ip for ip, h in env.global_state.all_hosts.items()
            if h.compromised_by == 'None' and h.status == 'online'
            and not ip.startswith('169.254.')
        ]
        if not targets:
            return np.array([0, 0], dtype=np.int64)
        ip = self._rng.choice(targets)
        return np.array(
            [0, target_ips.index(ip)],  # action 0 = ExploitRemoteService for red team
            dtype=np.int64,
        )
