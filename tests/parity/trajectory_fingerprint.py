"""Deterministic trajectory fingerprinting for cross-backend regression locking.

A "fingerprint" is a stable hash over a fixed-seed rollout of the environment.
The Phase-0 contract is: every later refactor (Phase 1 functional core, Phase 2
JAX backend, etc.) must reproduce the legacy fingerprint within a documented
tolerance, OR document the divergence as an intentional API change.

We intentionally hash only the *reward stream* and *terminal mask stream* —
not full observations — because:
  * The full obs contains floating-point SIEM/TF-IDF features sensitive to
    hash-seed drift (Python hash randomization), which would force PYTHONHASHSEED
    pinning on every consumer.
  * Rewards + termination capture the macro-dynamics that matter for RL parity:
    if these match, the agent sees the same training signal.

Float rewards are quantized to 6 decimals before hashing to absorb harmless
FP jitter across BLAS/MKL versions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from netforge_rl.environment.parallel_env import NetForgeRLEnv


REWARD_DECIMALS = 6


@dataclass
class Trajectory:
    rewards: list[dict[str, float]] = field(default_factory=list)
    terminations: list[dict[str, bool]] = field(default_factory=list)
    truncations: list[dict[str, bool]] = field(default_factory=list)
    step_count: int = 0

    def fingerprint(self) -> str:
        payload = {
            'rewards': [
                {k: round(float(v), REWARD_DECIMALS) for k, v in step.items()}
                for step in self.rewards
            ],
            'terminations': [
                {k: bool(v) for k, v in step.items()} for step in self.terminations
            ],
            'truncations': [
                {k: bool(v) for k, v in step.items()} for step in self.truncations
            ],
            'step_count': self.step_count,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(blob).hexdigest()


def _scripted_actions(env: NetForgeRLEnv, rng: np.random.Generator) -> dict:
    actions = {}
    for agent in env.agents:
        space = env.action_space(agent)
        actions[agent] = np.array(
            [rng.integers(0, n) for n in space.nvec], dtype=np.int64
        )
    return actions


def roll_trajectory(
    seed: int = 42,
    max_ticks: int = 50,
    scenario: str = 'ransomware',
) -> Trajectory:
    """Replays a deterministic episode and returns its trajectory record.

    The PRNG that drives actions is seeded explicitly so action choices are
    reproducible across machines; env.reset(seed=...) seeds the topology +
    physics.
    """
    config = {
        'scenario_type': scenario,
        'sim2real_mode': 'sim',
        'nlp_backend': 'tfidf',
        'max_ticks': max_ticks,
        'log_latency': 2,
    }
    env = NetForgeRLEnv(config)
    env.reset(seed=seed)

    rng = np.random.default_rng(seed)
    traj = Trajectory()

    while env.agents:
        actions = _scripted_actions(env, rng)
        _, rewards, term, trunc, _ = env.step(actions)
        traj.rewards.append(rewards)
        traj.terminations.append(term)
        traj.truncations.append(trunc)
        traj.step_count += 1
        if all(term.values()) or all(trunc.values()):
            break

    return traj
