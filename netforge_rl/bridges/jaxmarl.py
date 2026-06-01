"""JaxMARL-shape API on top of the JAX vector env.

Mirrors the ``jaxmarl.environments.MultiAgentEnv`` contract:

    reset(key) -> (obs: dict[agent, Array], state: JaxEnvState)
    step(key, state, actions: dict[agent, ...]) -> (obs, state, reward, done, info)

``obs`` is a per-agent dict whose values share the leading batch axis;
``actions`` accepts a dict-of-arrays or a structured BatchedActions
(internally upcast). Everything is jit-friendly: no Python dict
materialization inside the traced step — the dict mapping itself is
static, only its array values are traced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from netforge_rl.backends.jax import (
    BatchedActions,
    JaxEnvState,
    VectorEnvSpec,
    initial_batched_state,
    make_vector_step,
    to_jax,
)
from netforge_rl.core.functional import from_global_state
from netforge_rl.topologies.network_generator import NetworkGenerator


DEFAULT_AGENTS = ('red_operator', 'blue_dmz', 'blue_internal', 'blue_restricted')


def _per_agent_obs(state: JaxEnvState, agents: tuple[str, ...]) -> dict[str, jax.Array]:
    """Slice the batched state into one observation per agent.

    For Phase 3 every agent gets the same global view — the legacy env
    already shares state across blue agents — concatenated host arrays
    cast to float32. Per-role obs encoders land with the action port
    in Phase 2 slice 3.
    """
    flat = jnp.concatenate(
        [
            state.hosts.status.astype(jnp.float32),
            state.hosts.privilege.astype(jnp.float32),
            state.hosts.compromised_by_id.astype(jnp.float32),
            state.hosts.edr_active.astype(jnp.float32),
        ],
        axis=-1,
    )
    return {agent: flat for agent in agents}


@dataclass
class JaxMARLEnv:
    """JaxMARL-shape facade over :func:`make_vector_step`.

    The :attr:`batch_size` axis is exposed so callers can run thousands
    of envs at once. ``agents`` is the static agent list — both ``obs``
    and ``actions`` dicts key on it.
    """

    spec: VectorEnvSpec
    batch_size: int
    agents: tuple[str, ...] = DEFAULT_AGENTS

    def __post_init__(self) -> None:
        self._step = make_vector_step(self.spec)

    # ── lifecycle ─────────────────────────────────────────────────────

    def reset(self, key: jax.Array) -> tuple[dict[str, jax.Array], JaxEnvState]:
        """Generate a template state, tile to batch, return ``(obs, state)``."""
        seed = int(jax.random.randint(key, (), 0, 1 << 30))
        legacy = NetworkGenerator().generate(seed=seed)
        template = to_jax(from_global_state(legacy, agent_ids=self.agents))
        state = initial_batched_state(template, batch_size=self.batch_size)
        return _per_agent_obs(state, self.agents), state

    def step(
        self,
        key: jax.Array,
        state: JaxEnvState,
        actions: Mapping[str, jax.Array] | BatchedActions,
    ) -> tuple[
        dict[str, jax.Array],
        JaxEnvState,
        dict[str, jax.Array],
        dict[str, jax.Array],
        dict[str, jax.Array],
    ]:
        batched = self._coerce_actions(actions)
        new_state, rewards = self._step(state, batched)

        red_names = [a for a in self.agents if 'red' in a.lower()]
        blue_names = [a for a in self.agents if 'blue' in a.lower()]
        per_agent_reward: dict[str, jax.Array] = {}
        for i, name in enumerate(red_names):
            per_agent_reward[name] = rewards[:, i]
        for i, name in enumerate(blue_names):
            per_agent_reward[name] = rewards[:, self.spec.n_red + i]

        done = jnp.zeros((self.batch_size,), dtype=jnp.bool_)
        done_dict = {a: done for a in self.agents}
        info = {a: {} for a in self.agents}

        obs = _per_agent_obs(new_state, self.agents)
        return obs, new_state, per_agent_reward, done_dict, info

    # ── helpers ──────────────────────────────────────────────────────

    def _coerce_actions(self, actions) -> BatchedActions:
        if isinstance(actions, BatchedActions):
            return actions
        red_names = [a for a in self.agents if 'red' in a.lower()]
        blue_names = [a for a in self.agents if 'blue' in a.lower()]

        def stack(names) -> jax.Array:
            cols = [jnp.asarray(actions[name][..., 0], dtype=jnp.int32) for name in names]
            return jnp.stack(cols, axis=-1)

        def stack_attempt(names) -> jax.Array:
            cols = [
                jnp.asarray(actions[name][..., 1], dtype=jnp.bool_) for name in names
            ]
            return jnp.stack(cols, axis=-1)

        return BatchedActions(
            red_target_idx=stack(red_names),
            blue_target_idx=stack(blue_names),
            red_attempt=stack_attempt(red_names),
            blue_attempt=stack_attempt(blue_names),
        )


def random_action_dict(env: JaxMARLEnv, key: jax.Array) -> dict[str, jax.Array]:
    """Convenience sampler — one ``(target_idx, attempt_flag)`` per agent per env."""
    keys = jax.random.split(key, len(env.agents))
    out: dict[str, jax.Array] = {}
    for k, agent in zip(keys, env.agents):
        target = jax.random.randint(
            k, (env.batch_size,), 0, env.spec.n_hosts, dtype=jnp.int32
        )
        attempt = jax.random.bernoulli(k, p=0.5, shape=(env.batch_size,))
        out[agent] = jnp.stack(
            [target, attempt.astype(jnp.int32)], axis=-1
        )  # int32[B, 2]
    return out
