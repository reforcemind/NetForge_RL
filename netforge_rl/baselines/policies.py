import random
from typing import Protocol

import numpy as np


class BasePolicy(Protocol):
    name: str

    def act(self, env, agent_id: str) -> np.ndarray: ...


class RandomPolicy:
    """Uniform random over MultiDiscrete([32, 100])."""

    name = 'random'

    def __init__(self, seed=0):
        self._rng = np.random.default_rng(seed)

    def act(self, env, agent_id):
        space = env.action_space(agent_id)
        return np.array([self._rng.integers(0, n) for n in space.nvec], dtype=np.int64)


class HeuristicBluePolicy:
    """Blue: isolate the first compromised host; otherwise analyse host 0."""

    name = 'heuristic-blue'

    def __init__(self, seed=0):
        self._rng = random.Random(seed)

    def act(self, env, agent_id):
        target_ips = sorted(env.global_state.all_hosts.keys())
        compromised = [
            ip
            for ip, h in env.global_state.all_hosts.items()
            if h.compromised_by != 'None' and h.status != 'isolated'
        ]
        if compromised:
            ip = self._rng.choice(compromised)
            return np.array([0, target_ips.index(ip)], dtype=np.int64)
        return np.array([3, 0], dtype=np.int64)


class HeuristicRedPolicy:
    """Red: exploit the first uncompromised, non-padding online host."""

    name = 'heuristic-red'

    def __init__(self, seed=0):
        self._rng = random.Random(seed)

    def act(self, env, agent_id):
        target_ips = sorted(env.global_state.all_hosts.keys())
        targets = [
            ip
            for ip, h in env.global_state.all_hosts.items()
            if h.compromised_by == 'None'
            and h.status == 'online'
            and not ip.startswith('169.254.')
        ]
        if not targets:
            return np.array([0, 0], dtype=np.int64)
        ip = self._rng.choice(targets)
        return np.array([0, target_ips.index(ip)], dtype=np.int64)


_SCAN_SERVICES, _EXPLOIT, _PING_SWEEP = 2, 0, 15

class KillChainRedPolicy:
    """Red kill-chain: port-scan a reachable, vulnerable host then exploit it, expanding
    footholds DMZ -> Corporate -> Secure. Unlike HeuristicRedPolicy it recons first, so
    its exploits pass the prior-state check and actually compromise hosts."""

    name = 'killchain-red'

    def __init__(self, seed=0):
        self._rng = random.Random(seed)

    def act(self, env, agent_id):
        gs = env.global_state
        target_ips = sorted(gs.all_hosts.keys())
        history = gs.action_history.get(agent_id, set())
        candidates = [
            ip
            for ip in target_ips
            if gs.all_hosts[ip].compromised_by == 'None'
            and gs.all_hosts[ip].status == 'online'
            and gs.all_hosts[ip].vulnerabilities
            and not ip.startswith('169.254.')
            and gs.can_route_to(ip, agent_id=agent_id)
        ]
        if not candidates:
            return np.array([_PING_SWEEP, 0], dtype=np.int64)
        for ip in candidates:
            if f'DiscoverNetworkServices:{ip}' in history:
                return np.array([_EXPLOIT, target_ips.index(ip)], dtype=np.int64)
        ip = candidates[0]
        return np.array([_SCAN_SERVICES, target_ips.index(ip)], dtype=np.int64)
