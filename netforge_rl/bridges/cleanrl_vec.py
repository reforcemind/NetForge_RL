"""CleanRL-style synchronous vector env shim on top of :class:`JaxMARLEnv`.

Reduces the multi-agent JAX env to the single-agent shape that CleanRL,
Stable-Baselines3, and Tianshou consume:

    obs, info = vec.reset(seed)
    obs, reward, terminated, truncated, info = vec.step(action_int_array)

The wrapped agent is the policy under training. Other agents follow a
uniform-random policy by default — callers can swap in a callable for
``opponent_action_fn``.

Action space is a discrete ``n_hosts`` (the target index). The wrapped
agent always uses ``RED_COMPROMISE`` / ``BLUE_ISOLATE`` as its action
type since CleanRL expects a single discrete action. Richer action
types are reachable by constructing :class:`BatchedActions` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from netforge_rl.backends.jax import (
    BatchedActions,
    VectorEnvSpec,
)
from netforge_rl.bridges.jaxmarl import DEFAULT_AGENTS, JaxMARLEnv, random_action_dict


@dataclass
class CleanRLVecEnv:
    """Drop-in VecEnv for single-agent CleanRL / SB3 algorithms.

    Attributes:
        spec: shape contract.
        num_envs: parallel envs (the leading vmap axis).
        agent_id: which agent the wrapped policy controls.
        opponent_action_fn: produces a dict-of-arrays for the other agents.
            Defaults to uniform random.
    """

    spec: VectorEnvSpec
    num_envs: int
    agent_id: str = 'blue_dmz'
    opponent_action_fn: Callable[[JaxMARLEnv, jax.Array], dict] | None = None

    def __post_init__(self) -> None:
        self._env = JaxMARLEnv(
            spec=self.spec, batch_size=self.num_envs, agents=DEFAULT_AGENTS
        )
        self._state = None
        self._key = jax.random.PRNGKey(0)
        if self.opponent_action_fn is None:
            self.opponent_action_fn = random_action_dict

        # Gym-style spaces (advisory — CleanRL inspects these).
        self.single_action_space = ('discrete', self.spec.n_hosts)
        self.single_observation_space = self._obs_shape()

    def _obs_shape(self) -> tuple[int, ...]:
        return (4 * self.spec.n_hosts,)

    # ── Lifecycle ────────────────────────────────────────────────────

    def reset(self, seed: int = 0) -> tuple[np.ndarray, dict]:
        self._key = jax.random.PRNGKey(seed)
        self._key, sub = jax.random.split(self._key)
        obs, self._state = self._env.reset(sub)
        return np.asarray(obs[self.agent_id]), {}

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """``actions`` is int32[num_envs] — host target index for the controlled agent."""
        self._key, k_op, k_step = jax.random.split(self._key, 3)
        opponent_actions = self.opponent_action_fn(self._env, k_op)

        # Inject the controlled action — opponent_action_fn supplies
        # (target, attempt) pairs; we overwrite the controlled agent's
        # target and force attempt=True.
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
