"""Steps-Per-Second baseline harness for the legacy PyTorch/PettingZoo backend.

This locks in the pre-refactor throughput so every later phase can assert no
regression beyond a tolerated band. Run as:

    python -m benchmarks.sps_baseline --episodes 5 --max-ticks 1000
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from netforge_rl.environment.parallel_env import NetForgeRLEnv


@dataclass
class SPSResult:
    backend: str
    scenario: str
    episodes: int
    max_ticks: int
    total_steps: int
    wall_seconds: float
    sps_mean: float
    sps_median: float
    sps_min: float
    sps_max: float
    python: str
    platform: str


def _random_actions(env: NetForgeRLEnv, rng: np.random.Generator) -> dict:
    actions = {}
    for agent in env.agents:
        space = env.action_space(agent)
        actions[agent] = np.array(
            [rng.integers(0, n) for n in space.nvec], dtype=np.int64
        )
    return actions


def run(
    episodes: int = 5,
    max_ticks: int = 500,
    scenario: str = 'ransomware',
    seed: int = 42,
) -> SPSResult:
    config = {
        'scenario_type': scenario,
        'sim2real_mode': 'sim',
        'nlp_backend': 'tfidf',
        'max_ticks': max_ticks,
        'log_latency': 2,
    }
    env = NetForgeRLEnv(config)
    rng = np.random.default_rng(seed)

    per_ep_sps: list[float] = []
    total_steps = 0
    wall_start = time.perf_counter()

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        ep_steps = 0
        ep_start = time.perf_counter()
        while env.agents:
            actions = _random_actions(env, rng)
            _, _, term, trunc, _ = env.step(actions)
            ep_steps += 1
            if all(term.values()) or all(trunc.values()):
                break
        ep_wall = time.perf_counter() - ep_start
        per_ep_sps.append(ep_steps / ep_wall if ep_wall > 0 else 0.0)
        total_steps += ep_steps

    wall = time.perf_counter() - wall_start

    return SPSResult(
        backend='legacy-pytorch',
        scenario=scenario,
        episodes=episodes,
        max_ticks=max_ticks,
        total_steps=total_steps,
        wall_seconds=wall,
        sps_mean=statistics.mean(per_ep_sps),
        sps_median=statistics.median(per_ep_sps),
        sps_min=min(per_ep_sps),
        sps_max=max(per_ep_sps),
        python=platform.python_version(),
        platform=f'{platform.system()} {platform.release()}',
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--max-ticks', type=int, default=500)
    p.add_argument('--scenario', default='ransomware')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument(
        '--out',
        type=Path,
        default=Path('benchmarks/results/sps_baseline.json'),
    )
    args = p.parse_args()

    result = run(
        episodes=args.episodes,
        max_ticks=args.max_ticks,
        scenario=args.scenario,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(result), indent=2))
    print(json.dumps(asdict(result), indent=2))


if __name__ == '__main__':
    main()
