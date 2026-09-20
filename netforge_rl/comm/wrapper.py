from __future__ import annotations

from collections import deque

import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper


class CommFailureWrapper(BaseParallelWrapper):
    """Drop, delay, corrupt, or bandwidth-limit the Blue shared comm channel."""

    def __init__(
        self,
        env,
        drop_p: float = 0.0,
        delay_steps: int = 0,
        corrupt_p: float = 0.0,
        bandwidth: int | None = None,
        seed: int = 0,
    ):
        super().__init__(env)
        self.drop_p = float(drop_p)
        self.delay_steps = int(delay_steps)
        self.corrupt_p = float(corrupt_p)
        self.bandwidth = bandwidth
        self._rng = np.random.default_rng(seed)
        self._queue: deque[np.ndarray] = deque()

    def reset(self, seed=None, options=None):
        self._queue.clear()
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, infos = super().reset(seed=seed, options=options)
        return self._apply(obs), infos

    def step(self, actions):
        obs, rewards, term, trunc, infos = super().step(actions)
        return self._apply(obs), rewards, term, trunc, infos

    def _apply(self, obs: dict) -> dict:
        template = None
        for agent, payload in obs.items():
            if 'blue' in agent and isinstance(payload, dict) and 'blue_comm' in payload:
                template = np.array(payload['blue_comm'], dtype=np.float32, copy=True)
                break
        if template is None:
            return obs
        delayed = self._delay(template)
        degraded = self._degrade(delayed)
        for agent, payload in obs.items():
            if 'blue' in agent and isinstance(payload, dict) and 'blue_comm' in payload:
                payload['blue_comm'] = degraded
        return obs

    def _delay(self, comm: np.ndarray) -> np.ndarray:
        if self.delay_steps <= 0:
            return comm
        self._queue.append(comm)
        if len(self._queue) <= self.delay_steps:
            return np.zeros_like(comm)
        return self._queue.popleft()

    def _degrade(self, comm: np.ndarray) -> np.ndarray:
        out = comm.copy()
        if self.drop_p > 0 and float(self._rng.random()) < self.drop_p:
            return np.zeros_like(out)
        if self.corrupt_p > 0:
            mask = self._rng.random(out.shape) < self.corrupt_p
            out = np.where(mask, self._rng.random(out.shape, dtype=np.float32), out)
        if self.bandwidth is not None:
            k = max(0, int(self.bandwidth))
            if k < out.size:
                keep = np.argpartition(out, -k)[-k:] if k else []
                clipped = np.zeros_like(out)
                if k:
                    clipped[keep] = out[keep]
                out = clipped
        return out.astype(np.float32)
