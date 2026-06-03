from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from netforge_rl.backends.jax import VectorEnvSpec
from netforge_rl.bridges.jaxmarl import DEFAULT_AGENTS, JaxMARLEnv, random_action_dict


@dataclass
class CleanRLVecEnv:
    """Single-agent gym-5-tuple shim over JaxMARLEnv for CleanRL / SB3 / Tianshou."""

    spec: VectorEnvSpec
    num_envs: int
    agent_id: str = 'blue_dmz'
    opponent_action_fn: Callable | None = None

    def __post_init__(self):
        self._env = JaxMARLEnv(
            spec=self.spec, batch_size=self.num_envs, agents=DEFAULT_AGENTS
        )
        self._state = None
        self._key = jax.random.PRNGKey(0)
        if self.opponent_action_fn is None:
            self.opponent_action_fn = random_action_dict
        self.single_action_space = ('discrete', self.spec.n_hosts)
        self.single_observation_space = (4 * self.spec.n_hosts,)

    def reset(self, seed=0):
        self._key = jax.random.PRNGKey(seed)
        self._key, sub = jax.random.split(self._key)
        obs, self._state = self._env.reset(sub)
        return np.asarray(obs[self.agent_id]), {}

    def step(self, actions):
        """actions: int32[num_envs] target index for the controlled agent."""
        self._key, k_op, k_step = jax.random.split(self._key, 3)
        opponent_actions = self.opponent_action_fn(self._env, k_op)

        action_jax = jnp.asarray(actions, dtype=jnp.int32)
        opponent_actions[self.agent_id] = jnp.stack(
            [action_jax, jnp.ones_like(action_jax)], axis=-1
        )

        obs_dict, self._state, reward_dict, done_dict, info = self._env.step(
            k_step, self._state, opponent_actions
        )

        obs = np.asarray(obs_dict[self.agent_id])
        reward = np.asarray(reward_dict[self.agent_id], dtype=np.float32)
        terminated = np.asarray(done_dict[self.agent_id])
        truncated = np.zeros_like(terminated)
        return obs, reward, terminated, truncated, info
