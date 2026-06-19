import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax

from netforge_rl.backends.jax import VectorEnvSpec
from netforge_rl.baselines.jax_ppo import (
    PPOConfig,
    init_adam,
    init_mlp_params,
    make_rollout_scan,
)
from netforge_rl.bridges.jaxmarl import JaxMARLEnv


@dataclass
class Row:
    batch: int
    num_steps: int
    iters: int
    wall_seconds: float
    sec_per_iter: float
    rollout_sps: float


def measure(batch, num_steps, iters):
    cfg = PPOConfig(batch_size=batch, num_steps=num_steps, n_hosts=100)
    env = JaxMARLEnv(
        spec=VectorEnvSpec(n_hosts=cfg.n_hosts, n_red=cfg.n_red, n_blue=cfg.n_blue),
        batch_size=batch,
    )
    key = jax.random.PRNGKey(0)
    key, sub = jax.random.split(key)
    obs_dict, state = env.reset(sub)
    params = init_mlp_params(
        jax.random.PRNGKey(1), obs_dict[cfg.controlled_agent].shape[-1], cfg.n_hosts
    )
    _ = init_adam(params)
    rollout = make_rollout_scan(env, cfg)

    state, obs_dict, key, traj, last_value = rollout(params, state, obs_dict, key)
    traj['reward'].block_until_ready()

    t0 = time.perf_counter()
    for _ in range(iters):
        state, obs_dict, key, traj, last_value = rollout(
            params, state, obs_dict, key
        )
    traj['reward'].block_until_ready()
    wall = time.perf_counter() - t0

    total_steps = iters * num_steps * batch
    return Row(
        batch=batch,
        num_steps=num_steps,
        iters=iters,
        wall_seconds=wall,
        sec_per_iter=wall / iters,
        rollout_sps=total_steps / wall,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--num-steps', type=int, default=32)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument(
        '--out',
        type=Path,
        default=Path('benchmarks/results/ppo_rollout.json'),
    )
    args = p.parse_args()
    row = measure(args.batch, args.num_steps, args.iters)
    payload = {
        **asdict(row),
        'device': str(jax.devices()[0]),
        'python': platform.python_version(),
        'platform': f'{platform.system()} {platform.release()}',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
