from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / 'results'
N_AGENTS = 4


def bench_jax(batch_sizes, n_steps, warmup):
    import jax

    from netforge_rl.backends.jax import VectorEnvSpec, random_actions
    from netforge_rl.bridges.jaxmarl import JaxMARLEnv

    rows = []
    for batch in batch_sizes:
        spec = VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3)
        env = JaxMARLEnv(spec=spec, batch_size=batch)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        def run(state, key, steps):
            for _ in range(steps):
                key, sub = jax.random.split(key)
                actions = random_actions(spec, batch_size=batch, key=sub)
                _, state, _, _, _ = env.step(sub, state, actions)
            jax.block_until_ready(state.hosts.status)
            return state, key

        state, key = run(state, key, warmup)  # compile + warm
        t0 = time.perf_counter()
        state, key = run(state, key, n_steps)
        dt = time.perf_counter() - t0

        env_sps = (n_steps * batch) / dt
        rows.append(
            {
                'backend': 'jax',
                'batch_size': batch,
                'env_steps_per_sec': round(env_sps, 1),
                'agent_steps_per_sec': round(env_sps * N_AGENTS, 1),
                'wall_s': round(dt, 3),
            }
        )
        print(
            f'  jax   batch={batch:<5} {env_sps:>12,.0f} env-steps/s  '
            f'{env_sps * N_AGENTS:>13,.0f} agent-steps/s'
        )
    return rows


def bench_legacy(n_steps):
    from netforge_rl.environment.parallel_env import NetForgeRLEnv

    env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': n_steps + 10})
    env.reset(seed=0)
    rng = np.random.default_rng(0)

    def sample(env):
        return {
            a: np.array(
                [rng.integers(0, n) for n in env.action_space(a).nvec], dtype=np.int64
            )
            for a in env.agents
        }

    t0 = time.perf_counter()
    steps = 0
    while env.agents and steps < n_steps:
        env.step(sample(env))
        steps += 1
    dt = time.perf_counter() - t0
    env_sps = steps / dt
    print(
        f'  legacy batch=1     {env_sps:>12,.0f} env-steps/s  '
        f'{env_sps * N_AGENTS:>13,.0f} agent-steps/s'
    )
    return {
        'backend': 'legacy',
        'batch_size': 1,
        'env_steps_per_sec': round(env_sps, 1),
        'agent_steps_per_sec': round(env_sps * N_AGENTS, 1),
        'wall_s': round(dt, 3),
        'steps': steps,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 64, 256, 1024, 4096])
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--no-jax', action='store_true')
    p.add_argument('--no-legacy', action='store_true')
    args = p.parse_args()

    rows = []
    print('Throughput (one env-step = one tick across the batch):')
    if not args.no_jax:
        try:
            rows += bench_jax(args.batches, args.steps, args.warmup)
        except ImportError:
            print('  [skip] jax not installed')
    if not args.no_legacy:
        rows.append(bench_legacy(args.steps))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / 'throughput.json'
    out.write_text(json.dumps(rows, indent=2))
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
