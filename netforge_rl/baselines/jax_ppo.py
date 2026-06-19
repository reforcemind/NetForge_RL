from dataclasses import dataclass

import jax
import jax.numpy as jnp

from netforge_rl.backends.jax import BatchedActions, VectorEnvSpec
from netforge_rl.bridges.jaxmarl import JaxMARLEnv


def init_mlp_params(key, obs_dim, n_targets, n_action_types, hidden=64):
    """Two-layer MLP with separate policy + value heads."""
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    scale = 0.1
    return {
        'w1': jax.random.normal(k1, (obs_dim, hidden)) * scale,
        'b1': jnp.zeros((hidden,)),
        'w2': jax.random.normal(k2, (hidden, hidden)) * scale,
        'b2': jnp.zeros((hidden,)),
        'wp_target': jax.random.normal(k3, (hidden, n_targets)) * scale,
        'bp_target': jnp.zeros((n_targets,)),
        'wp_type': jax.random.normal(k4, (hidden, n_action_types)) * scale,
        'bp_type': jnp.zeros((n_action_types,)),
        'wv': jax.random.normal(k5, (hidden, 1)) * scale,
        'bv': jnp.zeros((1,)),
    }


def forward(params, obs):
    h = jax.nn.tanh(obs @ params['w1'] + params['b1'])
    h = jax.nn.tanh(h @ params['w2'] + params['b2'])
    logits_target = h @ params['wp_target'] + params['bp_target']
    logits_type = h @ params['wp_type'] + params['bp_type']
    value = (h @ params['wv'] + params['bv'])[..., 0]
    return (logits_target, logits_type), value


def init_adam(params, beta1=0.9, beta2=0.999):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return m, v, jnp.asarray(0, dtype=jnp.int32), beta1, beta2


def adam_update(params, grads, state, lr, eps=1e-8):
    m, v, t, b1, b2 = state
    t = t + 1
    m = jax.tree_util.tree_map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, grads)
    v = jax.tree_util.tree_map(lambda vi, gi: b2 * vi + (1 - b2) * gi * gi, v, grads)
    mhat = jax.tree_util.tree_map(lambda mi: mi / (1 - b1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda vi: vi / (1 - b2 ** t), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat
    )
    return new_params, (m, v, t, b1, b2)


def gae(rewards, values, dones, *, gamma, lam):
    """Generalized Advantage Estimation. values has length T+1."""
    def body(carry, t):
        gae_tm1 = carry
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae_t = delta + gamma * lam * (1.0 - dones[t]) * gae_tm1
        return gae_t, gae_t

    _, adv = jax.lax.scan(
        body, jnp.zeros_like(rewards[0]), jnp.arange(rewards.shape[0])[::-1]
    )
    return adv[::-1]


def ppo_loss(params, obs, actions, old_logp, advantages, returns, *, clip, vf_coef):
    (logits_target, logits_type), values = forward(params, obs)
    
    target_idx = actions[..., 0]
    action_type = actions[..., 1]
    
    logp_target_all = jax.nn.log_softmax(logits_target)
    logp_target = jnp.take_along_axis(logp_target_all, target_idx[..., None], axis=-1)[..., 0]
    
    logp_type_all = jax.nn.log_softmax(logits_type)
    logp_type = jnp.take_along_axis(logp_type_all, action_type[..., None], axis=-1)[..., 0]
    
    logp = logp_target + logp_type
    
    ratio = jnp.exp(logp - old_logp)
    pg = -jnp.minimum(
        ratio * advantages,
        jnp.clip(ratio, 1 - clip, 1 + clip) * advantages,
    ).mean()
    vf = 0.5 * ((values - returns) ** 2).mean()
    return pg + vf_coef * vf, (pg, vf)


@dataclass
class PPOConfig:
    total_iters: int = 5
    num_steps: int = 16
    batch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    n_hosts: int = 100
    n_action_types: int = 16
    n_red: int = 1
    n_blue: int = 3
    controlled_agent: str = 'blue_dmz'
    seed: int = 0
    update_epochs: int = 4
    num_minibatches: int = 4


def _build_actions(target_idx, action_type, env_agents, controlled, n_red, n_blue, n_hosts, key):
    """BatchedActions where controlled gets policy target; others uniform random."""
    k_rt, k_bt, k_rat, k_bat = jax.random.split(key, 4)
    red_t = jax.random.randint(k_rt, (target_idx.shape[0], n_red), 0, n_hosts, dtype=jnp.int32)
    blue_t = jax.random.randint(k_bt, (target_idx.shape[0], n_blue), 0, n_hosts, dtype=jnp.int32)
    red_at = jax.random.randint(k_rat, (target_idx.shape[0], n_red), 0, 16, dtype=jnp.int8)
    blue_at = jax.random.randint(k_bat, (target_idx.shape[0], n_blue), 0, 10, dtype=jnp.int8)

    red_names = [a for a in env_agents if 'red' in a.lower()]
    blue_names = [a for a in env_agents if 'blue' in a.lower()]
    if controlled in red_names:
        red_t = red_t.at[:, red_names.index(controlled)].set(target_idx)
        red_at = red_at.at[:, red_names.index(controlled)].set(action_type.astype(jnp.int8))
    elif controlled in blue_names:
        blue_t = blue_t.at[:, blue_names.index(controlled)].set(target_idx)
        blue_at = blue_at.at[:, blue_names.index(controlled)].set(action_type.astype(jnp.int8))

    return BatchedActions(
        red_target_idx=red_t,
        blue_target_idx=blue_t,
        red_attempt=jnp.ones((target_idx.shape[0], n_red), dtype=jnp.bool_),
        blue_attempt=jnp.ones((target_idx.shape[0], n_blue), dtype=jnp.bool_),
        red_action_type=red_at,
        blue_action_type=blue_at,
    )


def make_rollout_scan(env: JaxMARLEnv, cfg: PPOConfig):
    """Fuse cfg.num_steps env ticks into one jit'd XLA graph via lax.scan."""
    controlled = cfg.controlled_agent
    agents = env.agents
    n_red, n_blue, n_hosts = cfg.n_red, cfg.n_blue, cfg.n_hosts

    def step_body(carry, _):
        params, state, obs_dict, key = carry
        obs = obs_dict[controlled]
        (logits_target, logits_type), value = forward(params, obs)
        key, k_act_t, k_act_type, k_env = jax.random.split(key, 4)
        
        target_idx = jax.random.categorical(k_act_t, logits_target)
        action_type = jax.random.categorical(k_act_type, logits_type)
        
        logp_target_all = jax.nn.log_softmax(logits_target)
        logp_target = jnp.take_along_axis(logp_target_all, target_idx[..., None], axis=-1)[..., 0]
        
        logp_type_all = jax.nn.log_softmax(logits_type)
        logp_type = jnp.take_along_axis(logp_type_all, action_type[..., None], axis=-1)[..., 0]
        
        logp = logp_target + logp_type

        actions = _build_actions(
            target_idx, action_type, agents, controlled, n_red, n_blue, n_hosts, k_env
        )
        obs_dict, state, reward, done, _ = env.step(k_env, state, actions)
        out = {
            'obs': obs,
            'action': jnp.stack([target_idx, action_type], axis=-1),
            'logp': logp,
            'value': value,
            'reward': reward[controlled],
            'done': done[controlled].astype(jnp.float32),
        }
        return (params, state, obs_dict, key), out

    @jax.jit
    def rollout(params, state, obs_dict, key):
        (params, state, obs_dict, key), traj = jax.lax.scan(
            step_body, (params, state, obs_dict, key), None, length=cfg.num_steps
        )
        _, last_value = forward(params, obs_dict[controlled])
        return state, obs_dict, key, traj, last_value

    return rollout


def _make_grad_step(cfg: PPOConfig):
    @jax.jit
    def grad_step(params, opt_state, flat, idx):
        obs = flat['obs'][idx]
        action = flat['action'][idx]
        logp = flat['logp'][idx]
        adv = flat['adv'][idx]
        ret = flat['ret'][idx]
        (loss, _), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
            params, obs, action, logp, adv, ret,
            clip=cfg.clip, vf_coef=cfg.vf_coef,
        )
        params, opt_state = adam_update(params, grads, opt_state, lr=cfg.learning_rate)
        return params, opt_state, loss

    return grad_step


def train(cfg: PPOConfig):
    """Run PPO with a scan-fused rollout + minibatched updates."""
    total_samples = cfg.num_steps * cfg.batch_size
    if total_samples % cfg.num_minibatches != 0:
        raise ValueError(
            f'num_steps*batch_size ({total_samples}) must be divisible by '
            f'num_minibatches ({cfg.num_minibatches}).'
        )
    minibatch_size = total_samples // cfg.num_minibatches

    env = JaxMARLEnv(
        spec=VectorEnvSpec(n_hosts=cfg.n_hosts, n_red=cfg.n_red, n_blue=cfg.n_blue),
        batch_size=cfg.batch_size,
    )
    key = jax.random.PRNGKey(cfg.seed)
    key, sub = jax.random.split(key)
    obs_dict, state = env.reset(sub)
    obs_dim = obs_dict[cfg.controlled_agent].shape[-1]
    params = init_mlp_params(jax.random.PRNGKey(cfg.seed + 1), obs_dim, cfg.n_hosts, cfg.n_action_types)
    opt_state = init_adam(params)
    rollout = make_rollout_scan(env, cfg)
    grad_step = _make_grad_step(cfg)

    losses = []
    for _ in range(cfg.total_iters):
        state, obs_dict, key, traj, last_value = rollout(params, state, obs_dict, key)

        values_extended = jnp.concatenate([traj['value'], last_value[None]])
        advantages = gae(
            traj['reward'], values_extended, traj['done'], gamma=cfg.gamma, lam=cfg.lam
        )
        returns = advantages + values_extended[:-1]

        flat = {
            'obs': traj['obs'].reshape(-1, obs_dim),
            'action': traj['action'].reshape(-1, 2),
            'logp': traj['logp'].reshape(-1),
            'adv': advantages.reshape(-1),
            'ret': returns.reshape(-1),
        }
        flat['adv'] = (flat['adv'] - flat['adv'].mean()) / (flat['adv'].std() + 1e-8)

        iter_loss = 0.0
        n_updates = 0
        for _ in range(cfg.update_epochs):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, total_samples)
            for mb in range(cfg.num_minibatches):
                idx = perm[mb * minibatch_size : (mb + 1) * minibatch_size]
                params, opt_state, loss = grad_step(params, opt_state, flat, idx)
                iter_loss += float(loss)
                n_updates += 1
        losses.append(iter_loss / n_updates)

    return {'params': params, 'losses': losses}
