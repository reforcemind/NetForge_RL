"""Steps-Per-Second harness for the JAX vectorized backend.

Reports SPS across a sweep of batch sizes so the speedup curve vs the
legacy single-env baseline is easy to read. The compile time of the
first jit call is excluded from the measurement (we warm up with one
step, then time N steps).

Run:
    python -m benchmarks.sps_jax_vectorized --batches 1 16 128 1024 4096
"""

from __future__ import annotations

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
    rows: list[JaxSPSRow]
    python: str
    platform: str


def _setup_template_state():
    gen = NetworkGenerator()
    legacy = gen.generate(seed=42)
    snap = from_global_state(legacy, agent_ids=AGENTS)
    return to_jax(snap)


def measure(batch_size: int, steps: int = 100) -> JaxSPSRow:
    spec = VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3)
    template = _setup_template_state()
    state = initial_batched_state(template, batch_size=batch_size)
    step = make_vector_step(spec)

    key = jax.random.PRNGKey(0)

    # Warm up: compile + first dispatch.
    warmup_actions = random_actions(spec, batch_size, key)
    state, rewards = step(state, warmup_actions)
    rewards.block_until_ready()

    t0 = time.perf_counter()
    for i in range(steps):
        key, sub = jax.random.split(key)
        actions = random_actions(spec, batch_size, sub)
        state, rewards = step(state, actions)
    rewards.block_until_ready()  # force completion of the last step
    wall = time.perf_counter() - t0

    aggregate = (steps * batch_size) / wall
    per_env = steps / wall
    return JaxSPSRow(
        batch_size=batch_size,
        steps=steps,
        wall_seconds=wall,
        aggregate_sps=aggregate,
        per_env_sps=per_env,
    )


def run(batch_sizes: list[int], steps: int) -> JaxSPSResult:
    rows = [measure(b, steps=steps) for b in batch_sizes]
    return JaxSPSResult(
        backend='jax-vectorized',
        device=str(jax.devices()[0]),
        rows=rows,
        python=platform.python_version(),
        platform=f'{platform.system()} {platform.release()}',
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--batches',
        type=int,
        nargs='+',
        default=[1, 16, 128, 1024],
    )
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
            f'{row.batch_size:>8d} {row.aggregate_sps:>14.0f} '
            f'{row.per_env_sps:>14.1f}'
        )


if __name__ == '__main__':
    main()
