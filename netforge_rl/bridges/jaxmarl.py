from dataclasses import dataclass

import jax
import jax.numpy as jnp

from netforge_rl.backends.jax import (
    N_BLUE_ACTIONS,
    N_RED_ACTIONS,
    BatchedActions,
    VectorEnvSpec,
    initial_batched_state,
    make_vector_step,
    scenario_done,
    to_jax,
)
from netforge_rl.backends.jax.vector_env import jax_siem_features
from netforge_rl.core.functional import from_global_state
from netforge_rl.topologies.network_generator import NetworkGenerator


DEFAULT_AGENTS = ('red_operator', 'blue_dmz', 'blue_internal', 'blue_restricted')


def _per_agent_obs(state, agents, telemetry=False):
    """Per-role observation function.  Red agents see only recon'd hosts."""

    blue_parts = [
        state.hosts.status.astype(jnp.float32),
        state.hosts.privilege.astype(jnp.float32),
        state.hosts.compromised_by_id.astype(jnp.float32),
        state.hosts.edr_active.astype(jnp.float32),
    ]
    if telemetry:
        blue_parts.append(jax_siem_features(state.hosts))
    blue_flat = jnp.concatenate(blue_parts, axis=-1)

    known_mask = state.knowledge_mask[:, 0, :].astype(jnp.float32)
    red_status = state.hosts.status.astype(jnp.float32) * known_mask
    red_priv = state.hosts.privilege.astype(jnp.float32) * known_mask
    red_flat = jnp.concatenate([red_status, red_priv], axis=-1)

    obs = {}
    for agent in agents:
        if 'red' in agent.lower():
            obs[agent] = red_flat
        else:
            obs[agent] = blue_flat
    return obs


@dataclass
class JaxMARLEnv:
    """JaxMARL-shape facade over make_vector_step."""

    spec: VectorEnvSpec
    batch_size: int
    agents: tuple = DEFAULT_AGENTS
    evaluation_mode: bool = False
    telemetry_obs: bool = False

    def __post_init__(self):
        self._step = make_vector_step(self.spec)

    def reset(self, key):
        seed = int(jax.random.randint(key, (), 0, 1 << 30))
        legacy = NetworkGenerator(evaluation_mode=self.evaluation_mode).generate(
            seed=seed
        )
        template = to_jax(from_global_state(legacy, agent_ids=self.agents))
        state = initial_batched_state(template, batch_size=self.batch_size)
        return _per_agent_obs(state, self.agents, self.telemetry_obs), state

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

        terminated = scenario_done(new_state, self.spec)
        truncated = new_state.current_tick >= self.spec.horizon
        done = terminated | truncated
        done_dict = {a: done for a in self.agents}
        info = {
            a: {'terminated': terminated, 'truncated': truncated} for a in self.agents
        }

        obs = _per_agent_obs(new_state, self.agents, self.telemetry_obs)
        return obs, new_state, per_agent_reward, done_dict, info

    def _coerce_actions(self, actions):
        if isinstance(actions, BatchedActions):
            return actions
        red_names = [a for a in self.agents if 'red' in a.lower()]
        blue_names = [a for a in self.agents if 'blue' in a.lower()]

        def stack(names, idx, dtype):
            return jnp.stack(
                [jnp.asarray(actions[name][..., idx], dtype=dtype) for name in names],
                axis=-1,
            )

        return BatchedActions(
            red_target_idx=stack(red_names, 0, jnp.int32),
            blue_target_idx=stack(blue_names, 0, jnp.int32),
            red_attempt=stack(red_names, 1, jnp.bool_),
            blue_attempt=stack(blue_names, 1, jnp.bool_),
            red_action_type=stack(red_names, 2, jnp.int8),
            blue_action_type=stack(blue_names, 2, jnp.int8),
        )


def random_action_dict(env, key):
    """One (target_idx, attempt_flag) int32[B, 2] per agent."""
    keys = jax.random.split(key, len(env.agents))
    out = {}
    for k, agent in zip(keys, env.agents):
        k1, k2, k3 = jax.random.split(k, 3)
        target = jax.random.randint(
            k1, (env.batch_size,), 0, env.spec.n_hosts, dtype=jnp.int32
        )
        attempt = jax.random.bernoulli(k2, p=0.5, shape=(env.batch_size,))
        n_act = N_RED_ACTIONS if 'red' in agent.lower() else N_BLUE_ACTIONS
        action_type = jax.random.randint(
            k3, (env.batch_size,), 0, n_act, dtype=jnp.int32
        )
        out[agent] = jnp.stack(
            [target, attempt.astype(jnp.int32), action_type], axis=-1
        )
    return out
