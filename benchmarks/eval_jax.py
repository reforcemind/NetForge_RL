from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from netforge_rl.backends.jax import SCENARIO_IDS, VectorEnvSpec, random_actions
from netforge_rl.backends.jax.vector_env import BLUE_ISOLATE
from netforge_rl.baselines.jax_ppo import PPOConfig, forward, ippo_train
from netforge_rl.bridges.jaxmarl import DEFAULT_AGENTS, JaxMARLEnv

RESULTS_DIR = Path(__file__).parent / 'results'
BLUE = [a for a in DEFAULT_AGENTS if 'blue' in a]
RED = [a for a in DEFAULT_AGENTS if 'red' in a]


def _blue_actions(env, target, attempt, atype, key):
    """Assemble an action dict: blue gets (target, attempt, atype); red random."""
    out = random_actions(
        env.spec, batch_size=env.batch_size, key=key
    )  # BatchedActions for red defaults
    actions = {}
    for i, a in enumerate(RED):
        actions[a] = jnp.stack(
            [
                out.red_target_idx[:, i],
                out.red_attempt[:, i].astype(jnp.int32),
                out.red_action_type[:, i].astype(jnp.int32),
            ],
            axis=-1,
        )
    for a in BLUE:
        actions[a] = jnp.stack([target, attempt.astype(jnp.int32), atype], axis=-1)
    return actions


def policy_random(params, obs, state, env, key):
    from netforge_rl.bridges.jaxmarl import random_action_dict

    return random_action_dict(env, key)


def policy_heuristic(params, obs, state, env, key):
    """Blue isolates the first compromised host; red acts randomly."""
    compromised = state.hosts.compromised_by_id >= 0  # (B, n_hosts)
    any_comp = jnp.any(compromised, axis=-1)
    target = jnp.argmax(compromised.astype(jnp.int32), axis=-1).astype(jnp.int32)
    atype = jnp.full((env.batch_size,), BLUE_ISOLATE, dtype=jnp.int32)
    return _blue_actions(env, target, any_comp, atype, key)


def policy_ippo(params, obs, state, env, key):
    """Greedy IPPO: argmax target + action type from the shared policy."""
    blue_obs = obs[BLUE[0]]
    (logits_target, logits_type), _ = forward(params, blue_obs)
    target = jnp.argmax(logits_target, axis=-1).astype(jnp.int32)
    atype = jnp.argmax(logits_type, axis=-1).astype(jnp.int32)
    attempt = jnp.ones((env.batch_size,), dtype=jnp.bool_)
    return _blue_actions(env, target, attempt, atype, key)


def eval_policy(policy_fn, params, scenario, n_seeds, horizon, evaluation_mode):
    spec = VectorEnvSpec(
        n_hosts=100, n_red=1, n_blue=3, scenario=SCENARIO_IDS[scenario], horizon=horizon
    )
    env = JaxMARLEnv(spec=spec, batch_size=1, evaluation_mode=evaluation_mode)

    blue_rewards, final_comp = [], []
    for seed in range(n_seeds):
        key = jax.random.PRNGKey(seed)
        key, sub = jax.random.split(key)
        obs, state = env.reset(sub)
        ep_reward = 0.0
        for _ in range(horizon):
            key, ka, ks = jax.random.split(key, 3)
            actions = policy_fn(params, obs, state, env, ka)
            obs, state, reward, done, _ = env.step(ks, state, actions)
            ep_reward += float(np.mean([np.asarray(reward[a]) for a in BLUE]))
            if bool(np.asarray(done[BLUE[0]]).all()):
                break
        blue_rewards.append(ep_reward)
        comp = np.asarray(state.hosts.compromised_by_id >= 0).mean()
        final_comp.append(float(comp))

    r = np.array(blue_rewards)
    n = len(r)
    ci = 1.96 * float(r.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    return {
        'mean_blue_reward': round(float(r.mean()), 4),
        'ci95_blue_reward': round(ci, 4),
        'mean_final_compromised': round(float(np.mean(final_comp)), 4),
        'n_seeds': n,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', default='ransomware', choices=list(SCENARIO_IDS))
    p.add_argument('--iters', type=int, default=80)
    p.add_argument('--seeds', type=int, default=20)
    p.add_argument('--horizon', type=int, default=150)
    args = p.parse_args()

    print(f'Training IPPO on {args.scenario} ({args.iters} iters)...')
    cfg = PPOConfig(
        total_iters=args.iters,
        num_steps=32,
        batch_size=64,
        n_blue=3,
        scenario=SCENARIO_IDS[args.scenario],
    )
    trained = ippo_train(cfg)
    ippo_params = trained['params']

    policies = {
        'random': (policy_random, None),
        'heuristic': (policy_heuristic, None),
        'ippo': (policy_ippo, ippo_params),
    }

    rows = []
    print(f'\nEvaluating on {args.scenario}, {args.seeds} held-out-able seeds:')
    for name, (fn, params) in policies.items():
        m = eval_policy(fn, params, args.scenario, args.seeds, args.horizon, False)
        rows.append({'policy': name, 'scenario': args.scenario, **m})
        print(
            f'  {name:<12} blue_reward {m["mean_blue_reward"]:+8.3f} '
            f'± {m["ci95_blue_reward"]:.3f}   '
            f'final_comp {m["mean_final_compromised"]:.3f}'
        )

    # Generalization gap for the trained policy.
    train_m = eval_policy(
        policy_ippo, ippo_params, args.scenario, args.seeds, args.horizon, False
    )
    eval_m = eval_policy(
        policy_ippo, ippo_params, args.scenario, args.seeds, args.horizon, True
    )
    gap = round(train_m['mean_blue_reward'] - eval_m['mean_blue_reward'], 4)
    print(
        f'\n  IPPO generalization gap: train {train_m["mean_blue_reward"]:+.3f} '
        f'- held-out {eval_m["mean_blue_reward"]:+.3f} = {gap:+.3f}'
    )

    result = {
        'scenario': args.scenario,
        'leaderboard': sorted(rows, key=lambda x: x['mean_blue_reward'], reverse=True),
        'ippo_generalization': {
            'train': train_m['mean_blue_reward'],
            'held_out': eval_m['mean_blue_reward'],
            'gap': gap,
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f'jax_eval_{args.scenario}.json'
    out.write_text(json.dumps(result, indent=2))
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
