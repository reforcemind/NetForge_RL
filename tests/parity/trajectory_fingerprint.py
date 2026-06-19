import hashlib
import json
from dataclasses import dataclass, field
import numpy as np
from netforge_rl.environment.parallel_env import NetForgeRLEnv

REWARD_DECIMALS = 6


@dataclass
class Trajectory:
    rewards: list = field(default_factory=list)
    terminations: list = field(default_factory=list)
    truncations: list = field(default_factory=list)
    step_count: int = 0

    def fingerprint(self):
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


def _scripted_actions(env, rng):
    actions = {}
    for agent in env.agents:
        space = env.action_space(agent)
        actions[agent] = np.array(
            [rng.integers(0, n) for n in space.nvec], dtype=np.int64
        )
    return actions


def roll_trajectory(seed=42, max_ticks=50, scenario='ransomware'):
    config = {
        'scenario_type': scenario,
        'docker_bridge_mode': 'sim',
        'nlp_backend': 'tfidf',
        'max_ticks': max_ticks,
        'log_latency': 2,
    }
    env = NetForgeRLEnv(config)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    traj = Trajectory()
    while env.agents:
        _, rewards, term, trunc, _ = env.step(_scripted_actions(env, rng))
        traj.rewards.append(rewards)
        traj.terminations.append(term)
        traj.truncations.append(trunc)
        traj.step_count += 1
        if all(term.values()) or all(trunc.values()):
            break
    return traj
