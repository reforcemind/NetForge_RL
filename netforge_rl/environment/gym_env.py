from __future__ import annotations

import random as _random
from typing import Optional

import gymnasium as gym
import numpy as np

from netforge_rl.baselines.policies import RandomPolicy
from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.nlp.log_encoder import EMBEDDING_DIM

_FLAT_DIM = 256 + EMBEDDING_DIM


def _reseed(policy, seed: int) -> None:
    """Reseed a policy's RNG so opponents replay identically under a fixed seed."""
    rng = getattr(policy, '_rng', None)
    if isinstance(rng, _random.Random):
        policy._rng = _random.Random(seed)
    elif rng is not None:
        policy._rng = np.random.default_rng(seed)


class NetForgeSingleAgentEnv(gym.Env):
    """Gymnasium single-agent facade: one RL-controlled agent vs scripted opponents.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        scenario_type: str = 'ransomware',
        controlled_agent: str = 'blue_dmz',
        opponents: Optional[dict] = None,
        max_ticks: int = 200,
        config: Optional[dict] = None,
    ):
        super().__init__()
        cfg = dict(config or {})
        cfg.update(scenario_type=scenario_type, max_ticks=max_ticks)
        self._env = NetForgeRLEnv(cfg)
        self.controlled_agent = controlled_agent
        self._opponents = opponents or {}
        self._fallback = RandomPolicy(seed=0)
        self._last_flat = np.zeros(_FLAT_DIM, dtype=np.float32)

        self.action_space = self._env.action_space(controlled_agent)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(_FLAT_DIM,), dtype=np.float32
        )

    def _flat(self, agent_obs) -> np.ndarray:
        return np.concatenate([agent_obs['obs'], agent_obs['siem_embedding']]).astype(
            np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        base = seed or 0
        _reseed(self._fallback, base)
        for i, policy in enumerate(self._opponents.values()):
            _reseed(policy, base + i + 1)
        obs, infos = self._env.reset(seed=seed)
        a = self.controlled_agent
        self._last_flat = self._flat(obs[a])
        info = dict(infos.get(a, {}))
        info['action_mask'] = obs[a]['action_mask']
        return self._last_flat, info

    def step(self, action):
        actions = {}
        for agent in self._env.agents:
            if agent == self.controlled_agent:
                actions[agent] = np.asarray(action, dtype=np.int64)
            else:
                policy = self._opponents.get(agent, self._fallback)
                actions[agent] = policy.act(self._env, agent)

        obs, rewards, term, trunc, infos = self._env.step(actions)
        a = self.controlled_agent
        if a in obs:
            self._last_flat = self._flat(obs[a])
            info = dict(infos.get(a, {}))
            info['action_mask'] = obs[a]['action_mask']
            return (
                self._last_flat,
                float(rewards.get(a, 0.0)),
                bool(term.get(a, False)),
                bool(trunc.get(a, False)),
                info,
            )
        return self._last_flat, float(rewards.get(a, 0.0)), True, True, {}
