import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax

from netforge_rl.backends.jax import (
    VectorEnvSpec,
    initial_batched_state,
    make_vector_step,
    random_actions,
    to_jax,
)
from netforge_rl.core.functional import from_global_state
from netforge_rl.topologies.network_generator import NetworkGenerator


AGENTS = (
    'red_operator',
    'blue_dmz',
    'blue_internal',
    'blue_restricted',
)


@dataclass
class JaxSPSRow:
    batch_size: int
    steps: int
    wall_seconds: float
    aggregate_sps: float
    per_env_sps: float


@dataclass
class JaxSPSResult:
    backend: str
    device: str
    rows: list
    python: str
    platform: str


def _template_state():
    gen = NetworkGenerator()
    legacy = gen.generate(seed=42)
    snap = from_global_state(legacy, agent_ids=AGENTS)
    return to_jax(snap)


def measure(batch_size, steps=100):
    spec = VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3)
    state = initial_batched_state(_template_state(), batch_size=batch_size)
    step = make_vector_step(spec)
    key = jax.random.PRNGKey(0)

    state, rewards = step(state, random_actions(spec, batch_size, key))
    rewards.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(steps):
        key, sub = jax.random.split(key)
        state, rewards = step(state, random_actions(spec, batch_size, sub))
    rewards.block_until_ready()
    wall = time.perf_counter() - t0

    return JaxSPSRow(
        batch_size=batch_size,
        steps=steps,
        wall_seconds=wall,
        aggregate_sps=(steps * batch_size) / wall,
        per_env_sps=steps / wall,
    )


def run(batch_sizes, steps):
    rows = [measure(b, steps=steps) for b in batch_sizes]
    return JaxSPSResult(
        backend='jax-vectorized',
        device=str(jax.devices()[0]),
        rows=rows,
        python=platform.python_version(),
        platform=f'{platform.system()} {platform.release()}',
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 16, 128, 1024])
    p.add_argument('--steps', type=int, default=100)
    p.add_argument(
        '--out',
        type=Path,
        default=Path('benchmarks/results/sps_jax_vectorized.json'),
    )
    args = p.parse_args()

    result = run(args.batches, steps=args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(result), indent=2))

    print(json.dumps(asdict(result), indent=2))
    print('\n--- SPS summary ---')
    print(f'{"batch":>8} {"agg SPS":>14} {"per-env SPS":>14}')
    for row in result.rows:
        print(
            f'{row.batch_size:>8d} {row.aggregate_sps:>14.0f} {row.per_env_sps:>14.1f}'
        )


if __name__ == '__main__':
    main()
