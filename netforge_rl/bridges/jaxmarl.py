from dataclasses import dataclass

import jax
import jax.numpy as jnp

from netforge_rl.backends.jax import (
    BatchedActions,
    VectorEnvSpec,
    initial_batched_state,
    make_vector_step,
    to_jax,
)
from netforge_rl.core.functional import from_global_state
from netforge_rl.topologies.network_generator import NetworkGenerator


DEFAULT_AGENTS = ('red_operator', 'blue_dmz', 'blue_internal', 'blue_restricted')


def _per_agent_obs(state, agents):
    """Concatenated global view, broadcast to every agent for Phase 3."""
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
    """JaxMARL-shape facade over make_vector_step."""

    spec: VectorEnvSpec
    batch_size: int
    agents: tuple = DEFAULT_AGENTS

    def __post_init__(self):
        self._step = make_vector_step(self.spec)

    def reset(self, key):
        seed = int(jax.random.randint(key, (), 0, 1 << 30))
        legacy = NetworkGenerator().generate(seed=seed)
        template = to_jax(from_global_state(legacy, agent_ids=self.agents))
        state = initial_batched_state(template, batch_size=self.batch_size)
        return _per_agent_obs(state, self.agents), state

    def step(self, key, state, actions):
        batched = self._coerce_actions(actions)
        new_state, rewards = self._step(state, batched)

        red_names = [a for a in self.agents if 'red' in a.lower()]
        blue_names = [a for a in self.agents if 'blue' in a.lower()]
        per_agent_reward = {}
        for i, name in enumerate(red_names):
            per_agent_reward[name] = rewards[:, i]
        for i, name in enumerate(blue_names):
            per_agent_reward[name] = rewards[:, self.spec.n_red + i]

        done = jnp.zeros((self.batch_size,), dtype=jnp.bool_)
        done_dict = {a: done for a in self.agents}
        info = {a: {} for a in self.agents}

        obs = _per_agent_obs(new_state, self.agents)
        return obs, new_state, per_agent_reward, done_dict, info

    def _coerce_actions(self, actions):
        if isinstance(actions, BatchedActions):
            return actions
        red_names = [a for a in self.agents if 'red' in a.lower()]
        blue_names = [a for a in self.agents if 'blue' in a.lower()]

        def stack(names):
            return jnp.stack(
                [jnp.asarray(actions[name][..., 0], dtype=jnp.int32) for name in names],
                axis=-1,
            )

        def stack_attempt(names):
            return jnp.stack(
                [jnp.asarray(actions[name][..., 1], dtype=jnp.bool_) for name in names],
                axis=-1,
            )

        return BatchedActions(
            red_target_idx=stack(red_names),
            blue_target_idx=stack(blue_names),
            red_attempt=stack_attempt(red_names),
            blue_attempt=stack_attempt(blue_names),
        )


def random_action_dict(env, key):
    """One (target_idx, attempt_flag) int32[B, 2] per agent."""
    keys = jax.random.split(key, len(env.agents))
    out = {}
    for k, agent in zip(keys, env.agents):
        target = jax.random.randint(
            k, (env.batch_size,), 0, env.spec.n_hosts, dtype=jnp.int32
        )
        attempt = jax.random.bernoulli(k, p=0.5, shape=(env.batch_size,))
        out[agent] = jnp.stack([target, attempt.astype(jnp.int32)], axis=-1)
    return out
