"""Self-contained PureJaxRL-style PPO on top of :class:`JaxMARLEnv`.

Hand-rolled MLP + Adam so the only deps are JAX itself (avoids the
Windows long-path issue when pip installs the orbax tree). About 250
LOC end to end. Trains one shared actor-critic on the controlled
agent; other agents act randomly.

Key design choices that keep this jit-friendly:

  * Network params are a flat PyTree of arrays — no Flax module.
  * Adam state is a tuple ``(m, v, t)`` per param leaf.
  * The rollout loop uses :func:`jax.lax.scan` so the full
    ``num_steps``-deep loop unrolls inside one XLA graph.
  * Action sampling is a categorical over ``n_hosts`` target indices,
    with ``attempt=True`` always — the conflict resolver decides
    success. This gives one categorical head per controlled agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from netforge_rl.backends.jax import BatchedActions, VectorEnvSpec
from netforge_rl.bridges.jaxmarl import DEFAULT_AGENTS, JaxMARLEnv


# ── network ──────────────────────────────────────────────────────────────


def init_mlp_params(key: jax.Array, obs_dim: int, n_actions: int, hidden: int = 64):
    """Two-layer MLP with separate policy + value heads."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    scale = 0.1
    return {
        'w1': jax.random.normal(k1, (obs_dim, hidden)) * scale,
        'b1': jnp.zeros((hidden,)),
        'w2': jax.random.normal(k2, (hidden, hidden)) * scale,
        'b2': jnp.zeros((hidden,)),
        'wp': jax.random.normal(k3, (hidden, n_actions)) * scale,
        'bp': jnp.zeros((n_actions,)),
        'wv': jax.random.normal(k4, (hidden, 1)) * scale,
        'bv': jnp.zeros((1,)),
    }


def forward(params, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
    h = jax.nn.tanh(obs @ params['w1'] + params['b1'])
    h = jax.nn.tanh(h @ params['w2'] + params['b2'])
    logits = h @ params['wp'] + params['bp']
    value = (h @ params['wv'] + params['bv'])[..., 0]
    return logits, value


# ── Adam ─────────────────────────────────────────────────────────────────


def init_adam(params, beta1: float = 0.9, beta2: float = 0.999):
    return jax.tree_util.tree_map(jnp.zeros_like, params), jax.tree_util.tree_map(
        jnp.zeros_like, params
    ), jnp.asarray(0, dtype=jnp.int32), beta1, beta2


def adam_update(params, grads, state, lr: float, eps: float = 1e-8):
    m, v, t, b1, b2 = state
    t = t + 1
    m = jax.tree_util.tree_map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, grads)
    v = jax.tree_util.tree_map(
        lambda vi, gi: b2 * vi + (1 - b2) * gi * gi, v, grads
    )
    mhat = jax.tree_util.tree_map(lambda mi: mi / (1 - b1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda vi: vi / (1 - b2 ** t), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat
    )
    return new_params, (m, v, t, b1, b2)


# ── GAE + PPO loss ───────────────────────────────────────────────────────


def gae(rewards, values, dones, *, gamma: float, lam: float):
    """Generalized Advantage Estimation. ``values`` has length T+1."""
    def body(carry, t):
        gae_tm1 = carry
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae_t = delta + gamma * lam * (1.0 - dones[t]) * gae_tm1
        return gae_t, gae_t

    _, adv = jax.lax.scan(
        body, jnp.zeros_like(rewards[0]), jnp.arange(rewards.shape[0])[::-1]
    )
    return adv[::-1]


def ppo_loss(
    params, obs, actions, old_logp, advantages, returns, *, clip: float, vf_coef: float
):
    logits, values = forward(params, obs)
    logp_all = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(logp_all, actions[..., None], axis=-1)[..., 0]
    ratio = jnp.exp(logp - old_logp)
    pg = -jnp.minimum(
        ratio * advantages,
        jnp.clip(ratio, 1 - clip, 1 + clip) * advantages,
    ).mean()
    vf = 0.5 * ((values - returns) ** 2).mean()
    return pg + vf_coef * vf, (pg, vf)


# ── Trainer ─────────────────────────────────────────────────────────────


@dataclass
class PPOConfig:
    total_iters: int = 5
    num_steps: int = 16          # rollout horizon per iter
    batch_size: int = 32         # parallel envs
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    n_hosts: int = 100
    n_red: int = 1
    n_blue: int = 3
    controlled_agent: str = 'blue_dmz'
    seed: int = 0


def _build_actions(target_idx, env_agents, controlled, n_red, n_blue, n_hosts, key):
    """Construct a BatchedActions where the controlled agent's target comes from
    the policy and all others are uniformly random."""
    k_rt, k_bt, k_ra, k_ba = jax.random.split(key, 4)
    red_t = jax.random.randint(k_rt, (target_idx.shape[0], n_red), 0, n_hosts, dtype=jnp.int32)
    blue_t = jax.random.randint(k_bt, (target_idx.shape[0], n_blue), 0, n_hosts, dtype=jnp.int32)

    red_names = [a for a in env_agents if 'red' in a.lower()]
    blue_names = [a for a in env_agents if 'blue' in a.lower()]
    if controlled in red_names:
        red_t = red_t.at[:, red_names.index(controlled)].set(target_idx)
    elif controlled in blue_names:
        blue_t = blue_t.at[:, blue_names.index(controlled)].set(target_idx)

    return BatchedActions(
        red_target_idx=red_t,
        blue_target_idx=blue_t,
        red_attempt=jnp.ones((target_idx.shape[0], n_red), dtype=jnp.bool_),
        blue_attempt=jnp.ones((target_idx.shape[0], n_blue), dtype=jnp.bool_),
    )


def train(cfg: PPOConfig) -> dict:
    """Run PPO. Returns the final params + iteration loss trace.

    The full rollout loop is *not* placed inside jax.lax.scan in this
    reference impl — that's the obvious follow-up perf win once a
    Phase 5 scenario suite lands. Iteration count is intentionally low
    so the smoke test stays cheap on CI.
    """
    env = JaxMARLEnv(
        spec=VectorEnvSpec(n_hosts=cfg.n_hosts, n_red=cfg.n_red, n_blue=cfg.n_blue),
        batch_size=cfg.batch_size,
    )
    key = jax.random.PRNGKey(cfg.seed)
    key, sub = jax.random.split(key)
    obs_dict, state = env.reset(sub)
    obs_dim = obs_dict[cfg.controlled_agent].shape[-1]
    params = init_mlp_params(jax.random.PRNGKey(cfg.seed + 1), obs_dim, cfg.n_hosts)
    opt_state = init_adam(params)

    losses: list[float] = []
    for it in range(cfg.total_iters):
        # Rollout
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        for _ in range(cfg.num_steps):
            obs = obs_dict[cfg.controlled_agent]
            logits, value = forward(params, obs)
            key, k_act, k_env = jax.random.split(key, 3)
            target_idx = jax.random.categorical(k_act, logits)
            logp_all = jax.nn.log_softmax(logits)
            logp = jnp.take_along_axis(logp_all, target_idx[..., None], axis=-1)[..., 0]

            actions = _build_actions(
                target_idx, env.agents, cfg.controlled_agent,
                cfg.n_red, cfg.n_blue, cfg.n_hosts, k_env,
            )
            obs_dict, state, reward, done, _ = env.step(k_env, state, actions)

            obs_buf.append(obs)
            act_buf.append(target_idx)
            logp_buf.append(logp)
            val_buf.append(value)
            rew_buf.append(reward[cfg.controlled_agent])
            done_buf.append(done[cfg.controlled_agent].astype(jnp.float32))

        # Bootstrap value of last obs.
        _, last_value = forward(params, obs_dict[cfg.controlled_agent])
        values_extended = jnp.stack(val_buf + [last_value])
        rewards = jnp.stack(rew_buf)
        dones = jnp.stack(done_buf)
        advantages = gae(rewards, values_extended, dones, gamma=cfg.gamma, lam=cfg.lam)
        returns = advantages + values_extended[:-1]

        obs_arr = jnp.stack(obs_buf).reshape(-1, obs_dim)
        act_arr = jnp.stack(act_buf).reshape(-1)
        logp_arr = jnp.stack(logp_buf).reshape(-1)
        adv_arr = advantages.reshape(-1)
        ret_arr = returns.reshape(-1)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        (loss, _), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
            params, obs_arr, act_arr, logp_arr, adv_arr, ret_arr,
            clip=cfg.clip, vf_coef=cfg.vf_coef,
        )
        params, opt_state = adam_update(params, grads, opt_state, lr=cfg.learning_rate)
        losses.append(float(loss))

    return {'params': params, 'losses': losses}
