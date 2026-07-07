from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Protocol, runtime_checkable

import numpy as np

from netforge_rl.baselines.policies import HeuristicBluePolicy, HeuristicRedPolicy
from netforge_rl.environment.parallel_env import NetForgeRLEnv

LEADERBOARD_PATH = Path(__file__).parent / 'results' / 'leaderboard.json'
_ACT_TIMEOUT_S = 1.0


@runtime_checkable
class Agent(Protocol):
    def reset(self) -> None: ...
    def act(self, obs: dict, agent_id: str) -> np.ndarray: ...


class RandomAgent:
    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: dict, agent_id: str) -> np.ndarray:
        mask = obs.get('action_mask')
        if mask is not None:
            valid_types = np.where(mask[:32])[0]
            valid_targets = np.where(mask[32:])[0]
            t = int(self._rng.choice(valid_types)) if len(valid_types) else 0
            h = int(self._rng.choice(valid_targets)) if len(valid_targets) else 0
            return np.array([t, h], dtype=np.int64)
        return np.array([0, 0], dtype=np.int64)


class PolicyAgent:
    """Adapts a ``BasePolicy`` (which reads the live env) to the Agent protocol."""

    def __init__(self, policy):
        self._policy = policy
        self._env = None

    def bind_env(self, env) -> None:
        self._env = env

    def reset(self) -> None:
        pass

    def act(self, obs: dict, agent_id: str) -> np.ndarray:
        return np.asarray(self._policy.act(self._env, agent_id), dtype=np.int64)


@dataclass
class EpisodeResult:
    episode_id: str
    scenario: str
    seed: int
    steps: int
    red_total_reward: float
    blue_total_reward: float
    compromised_hosts: int
    isolated_hosts: int
    sla_uptime: float
    mttc: float
    total_exfiltrated: float
    red_agent_errors: int
    blue_agent_errors: int
    wall_time_s: float


@dataclass
class SubmissionResult:
    name: str
    team: str  # 'red' or 'blue'
    episodes: List[EpisodeResult] = field(default_factory=list)

    def aggregate(self) -> dict:
        if not self.episodes:
            return {'score': 0.0, 'episodes': 0}
        n = len(self.episodes)
        mean_comp = float(np.mean([e.compromised_hosts for e in self.episodes]))
        mean_sla = float(np.mean([e.sla_uptime for e in self.episodes]))
        mean_mttc = float(np.mean([e.mttc for e in self.episodes]))
        mean_exfil = float(np.mean([e.total_exfiltrated for e in self.episodes]))
        mean_red_r = float(np.mean([e.red_total_reward for e in self.episodes]))
        mean_blue_r = float(np.mean([e.blue_total_reward for e in self.episodes]))
        error_rate = float(
            sum(e.red_agent_errors + e.blue_agent_errors for e in self.episodes) / n
        )
        if self.team == 'red':
            score = (
                mean_comp * 2.0 + mean_exfil * 0.01 + mean_red_r * 0.1 - mean_sla * 10.0
            )
        else:
            score = (
                mean_sla * 50.0 - mean_comp * 2.0 - mean_mttc * 0.1 + mean_blue_r * 0.1
            )
        return {
            'score': round(score, 4),
            'episodes': n,
            'mean_red_reward': round(mean_red_r, 4),
            'mean_blue_reward': round(mean_blue_r, 4),
            'mean_compromised': round(mean_comp, 2),
            'mean_sla_uptime': round(mean_sla, 4),
            'mean_mttc': round(mean_mttc, 2),
            'mean_exfiltrated': round(mean_exfil, 2),
            'error_rate': round(error_rate, 4),
        }


def _safe_act(agent: Agent, obs: dict, agent_id: str) -> tuple[np.ndarray, bool]:
    try:
        return np.asarray(agent.act(obs, agent_id), dtype=np.int64), False
    except Exception:
        traceback.print_exc()
        return np.array([0, 0], dtype=np.int64), True


def run_episode(
    red_agent: Agent,
    blue_agent: Agent,
    scenario: str = 'ransomware',
    seed: int = 0,
    max_ticks: int = 200,
    evaluation_mode: bool = False,
) -> EpisodeResult:
    env = NetForgeRLEnv(
        {
            'scenario_type': scenario,
            'docker_mode': 'sim',
            'nlp_backend': 'tfidf',
            'max_ticks': max_ticks,
            'evaluation_mode': evaluation_mode,
        }
    )
    obs_dict, _ = env.reset(seed=seed)
    for agent in (red_agent, blue_agent):
        if hasattr(agent, 'bind_env'):
            agent.bind_env(env)
    red_agent.reset()
    blue_agent.reset()

    red_reward_total = blue_reward_total = 0.0
    red_errors = blue_errors = 0
    last_info: dict = {}
    t0 = time.perf_counter()

    while env.agents:
        actions = {}
        for agent_id in env.agents:
            obs = obs_dict.get(agent_id, {})
            if 'red' in agent_id:
                act, err = _safe_act(red_agent, obs, agent_id)
                red_errors += int(err)
            else:
                act, err = _safe_act(blue_agent, obs, agent_id)
                blue_errors += int(err)
            actions[agent_id] = act

        obs_dict, rewards, term, trunc, last_info = env.step(actions)
        for agent_id, r in rewards.items():
            if 'red' in agent_id:
                red_reward_total += float(r)
            else:
                blue_reward_total += float(r)

    sample_info = next((v for k, v in last_info.items() if 'blue' in k), {})
    return EpisodeResult(
        episode_id=str(uuid.uuid4())[:8],
        scenario=scenario,
        seed=seed,
        steps=env.current_tick,
        red_total_reward=red_reward_total,
        blue_total_reward=blue_reward_total,
        compromised_hosts=int(sample_info.get('compromised_hosts', 0)),
        isolated_hosts=int(sample_info.get('isolated_hosts', 0)),
        sla_uptime=float(sample_info.get('SLA_Uptime_Percentage', 1.0)),
        mttc=float(sample_info.get('MTTC', 0.0)),
        total_exfiltrated=float(sample_info.get('Total_Exfiltrated_Data', 0.0)),
        red_agent_errors=red_errors,
        blue_agent_errors=blue_errors,
        wall_time_s=time.perf_counter() - t0,
    )


def evaluate(
    submission: SubmissionResult,
    red_agent: Agent,
    blue_agent: Agent,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    max_ticks: int = 200,
) -> SubmissionResult:
    scenarios = scenarios or ['ransomware', 'apt_espionage']
    seeds = seeds or list(range(5))
    total = len(scenarios) * len(seeds)
    done = 0
    for scenario in scenarios:
        for seed in seeds:
            ep = run_episode(
                red_agent, blue_agent, scenario=scenario, seed=seed, max_ticks=max_ticks
            )
            submission.episodes.append(ep)
            done += 1
            print(
                f'  [{done}/{total}] {scenario} seed={seed} '
                f'red_r={ep.red_total_reward:+.1f} blue_r={ep.blue_total_reward:+.1f} '
                f'comp={ep.compromised_hosts} sla={ep.sla_uptime:.2f} '
                f'wall={ep.wall_time_s:.1f}s'
            )
    return submission


def load_leaderboard() -> list:
    if LEADERBOARD_PATH.exists():
        return json.loads(LEADERBOARD_PATH.read_text())
    return []


def save_leaderboard(entries: list) -> None:
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_PATH.write_text(json.dumps(entries, indent=2))


def submit_to_leaderboard(result: SubmissionResult) -> None:
    """Persist result and maintain separate red/blue leaderboard tables."""
    entries = load_leaderboard()
    agg = result.aggregate()
    entry = {
        'name': result.name,
        'team': result.team,
        'submitted_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        **agg,
    }
    entries = [
        e
        for e in entries
        if not (e['name'] == result.name and e['team'] == result.team)
    ]
    entries.append(entry)
    red_entries = sorted(
        [e for e in entries if e.get('team') == 'red'],
        key=lambda e: e['score'],
        reverse=True,
    )
    blue_entries = sorted(
        [e for e in entries if e.get('team') == 'blue'],
        key=lambda e: e['score'],
        reverse=True,
    )
    for rank, e in enumerate(red_entries, 1):
        e['team_rank'] = rank
    for rank, e in enumerate(blue_entries, 1):
        e['team_rank'] = rank
    entries = red_entries + blue_entries
    save_leaderboard(entries)
    print(f'\nLeaderboard saved: {LEADERBOARD_PATH}')


def print_leaderboard() -> None:
    entries = load_leaderboard()
    if not entries:
        print('Leaderboard is empty.')
        return
    for team_label, team_key in [('RED', 'red'), ('BLUE', 'blue')]:
        team_entries = [e for e in entries if e.get('team') == team_key]
        if not team_entries:
            continue
        print(f'\n=== {team_label} TEAM LEADERBOARD ===')
        hdr = f'{"Rank":<5} {"Name":<24} {"Score":>8} {"SLA%":>7} {"Comp":>6} {"MTTC":>7} {"Errors":>7}'
        print(hdr)
        print('-' * len(hdr))
        for e in sorted(team_entries, key=lambda x: x.get('team_rank', 999)):
            print(
                f'{e.get("team_rank", "-"):<5} {e["name"]:<24} '
                f'{e["score"]:>8.2f} {e.get("mean_sla_uptime", 0) * 100:>6.1f}% '
                f'{e.get("mean_compromised", 0):>6.1f} {e.get("mean_mttc", 0):>7.1f} '
                f'{e.get("error_rate", 0):>7.3f}'
            )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NetForge RL Competition Evaluator')
    parser.add_argument(
        '--leaderboard', action='store_true', help='Print current leaderboard'
    )
    parser.add_argument('--name', default='heuristic_baseline', help='Submission name')
    parser.add_argument('--team', default='red', choices=['red', 'blue'])
    parser.add_argument('--episodes', type=int, default=5, help='Episodes per scenario')
    parser.add_argument('--max-ticks', type=int, default=200)
    parser.add_argument(
        '--scenarios', nargs='+', default=['ransomware', 'apt_espionage']
    )
    args = parser.parse_args()

    if args.leaderboard:
        print_leaderboard()
    else:
        # The evaluated team plays its heuristic policy; the opponent does too.
        red_agent = PolicyAgent(HeuristicRedPolicy(seed=42))
        blue_agent = PolicyAgent(HeuristicBluePolicy(seed=42))
        print(f'Evaluating "{args.name}" ({args.team}) against a heuristic opponent...')
        sub = SubmissionResult(name=args.name, team=args.team)
        evaluate(
            sub,
            red_agent=red_agent,
            blue_agent=blue_agent,
            scenarios=args.scenarios,
            seeds=list(range(args.episodes)),
            max_ticks=args.max_ticks,
        )
        agg = sub.aggregate()
        print(
            f'\nScore: {agg["score"]:.4f}  (SLA {agg["mean_sla_uptime"] * 100:.1f}%  '
            f'comp {agg["mean_compromised"]:.1f}  mttc {agg["mean_mttc"]:.1f})'
        )
        submit_to_leaderboard(sub)
        print_leaderboard()
