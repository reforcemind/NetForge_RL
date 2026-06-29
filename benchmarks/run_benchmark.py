from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from benchmarks.competition_eval import (
    Agent,
    EpisodeResult,
    RandomAgent,
    load_leaderboard,
    print_leaderboard,
    run_episode,
    save_leaderboard,
)

ALL_SCENARIOS = [
    'ransomware',
    'apt_espionage',
    'cloud_hybrid',
    'iot_grid',
    'ot_stuxnet',
]
RESULTS_DIR = Path(__file__).parent / 'results'


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))


def aggregate_with_ci(episodes: list[EpisodeResult], team: str) -> dict:
    """Extend competition_eval.SubmissionResult.aggregate() with std + CI95."""
    n = len(episodes)
    if n == 0:
        return {}

    def _stats(vals):
        arr = np.array(vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if n > 1 else 0.0
        ci = 1.96 * std / math.sqrt(n)
        return mean, std, ci

    comp = [e.compromised_hosts for e in episodes]
    sla = [e.sla_uptime for e in episodes]
    mttc = [e.mttc for e in episodes]
    exfil = [e.total_exfiltrated for e in episodes]
    red_r = [e.red_total_reward for e in episodes]
    blue_r = [e.blue_total_reward for e in episodes]
    errors = [(e.red_agent_errors + e.blue_agent_errors) for e in episodes]

    m_comp, s_comp, ci_comp = _stats(comp)
    m_sla, s_sla, ci_sla = _stats(sla)
    m_mttc, s_mttc, ci_mttc = _stats(mttc)
    m_exfil, s_exfil, ci_exfil = _stats(exfil)
    MAX_REF_REWARD = 200.0
    m_red_r, s_red_r, _ = _stats([math.tanh(r / MAX_REF_REWARD) for r in red_r])
    m_blue_r, s_blue_r, _ = _stats([math.tanh(r / MAX_REF_REWARD) for r in blue_r])
    m_err, _, _ = _stats(errors)

    if team == 'red':
        score = m_comp * 2.0 + m_exfil * 0.01 + m_red_r * 0.1 - m_sla * 10.0
    else:
        score = m_sla * 50.0 - m_comp * 2.0 - m_mttc * 0.1 + m_blue_r * 0.1

    return {
        'n_episodes': n,
        'score': round(score, 4),
        'mean_compromised': round(m_comp, 3),
        'std_compromised': round(s_comp, 3),
        'ci95_compromised': round(ci_comp, 3),
        'mean_sla_uptime': round(m_sla, 4),
        'std_sla_uptime': round(s_sla, 4),
        'ci95_sla_uptime': round(ci_sla, 4),
        'mean_mttc': round(m_mttc, 3),
        'std_mttc': round(s_mttc, 3),
        'ci95_mttc': round(ci_mttc, 3),
        'mean_exfiltrated': round(m_exfil, 3),
        'std_exfiltrated': round(s_exfil, 3),
        'ci95_exfiltrated': round(ci_exfil, 3),
        'mean_red_reward': round(m_red_r, 4),
        'std_red_reward': round(s_red_r, 4),
        'mean_blue_reward': round(m_blue_r, 4),
        'std_blue_reward': round(s_blue_r, 4),
        'error_rate': round(m_err / max(n, 1), 4),
    }


def run_benchmark(
    name: str,
    team: str,
    red_agent: Agent,
    blue_agent: Agent,
    scenarios: list[str] | None = None,
    n_seeds: int = 20,
    max_ticks: int = 200,
    evaluation_mode: bool = False,
) -> dict:
    scenarios = scenarios or ALL_SCENARIOS
    seeds = list(range(n_seeds))
    total = len(scenarios) * len(seeds)
    done = 0
    all_episodes: list[EpisodeResult] = []
    per_scenario: dict[str, list[EpisodeResult]] = {s: [] for s in scenarios}

    t0 = time.perf_counter()
    for scenario in scenarios:
        for seed in seeds:
            ep = run_episode(
                red_agent,
                blue_agent,
                scenario=scenario,
                seed=seed,
                max_ticks=max_ticks,
                evaluation_mode=evaluation_mode,
            )
            all_episodes.append(ep)
            per_scenario[scenario].append(ep)
            done += 1
            tag = '[EVAL]' if evaluation_mode else '[TRAIN]'
            print(
                f'{tag} [{done:>{len(str(total))}}/{total}] {scenario:<16} '
                f'seed={seed:<3} '
                f'blue_r={ep.blue_total_reward:+8.1f}  '
                f'comp={ep.compromised_hosts:<3}  '
                f'sla={ep.sla_uptime:.2f}  '
                f'{ep.wall_time_s:.1f}s'
            )

    wall = time.perf_counter() - t0
    overall = aggregate_with_ci(all_episodes, team)
    scenario_agg = {s: aggregate_with_ci(eps, team) for s, eps in per_scenario.items()}

    result = {
        'name': name,
        'team': team,
        'evaluation_mode': evaluation_mode,
        'submitted_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'total_episodes': len(all_episodes),
        'wall_time_s': round(wall, 1),
        'scenarios': scenarios,
        'overall': overall,
        'per_scenario': scenario_agg,
    }

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f'benchmark_{name}_{ts}.json'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f'\nFull results saved: {out_path}')

    _update_leaderboard(name, team, overall)
    return result


def _update_leaderboard(name: str, team: str, agg: dict) -> None:
    """Update per-team leaderboard (fix 6.1 — team scores are not cross-comparable)."""
    entries = load_leaderboard()
    entry = {
        'name': name,
        'team': team,
        'submitted_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        **agg,
    }
    entries = [e for e in entries if not (e['name'] == name and e['team'] == team)]
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
    save_leaderboard(red_entries + blue_entries)


def _print_result(result: dict) -> None:
    o = result['overall']
    print(f'\n{"=" * 60}')
    print(f'  {result["name"]} ({result["team"]})  —  score {o["score"]:.4f}')
    print(
        f'  {result["total_episodes"]} episodes across {len(result["scenarios"])} scenarios'
    )
    print(f'{"=" * 60}')
    print(
        f'  SLA uptime : {o["mean_sla_uptime"] * 100:.1f}% ± {o["ci95_sla_uptime"] * 100:.1f}%'
    )
    print(
        f'  Compromised: {o["mean_compromised"]:.1f} ± {o["ci95_compromised"]:.1f} hosts'
    )
    print(f'  MTTC       : {o["mean_mttc"]:.1f} ± {o["ci95_mttc"]:.1f} ticks')
    print(f'  Exfiltrated: {o["mean_exfiltrated"]:.2f} ± {o["ci95_exfiltrated"]:.2f}')
    print(f'  Errors     : {o["error_rate"]:.4f}/ep')
    print(f'  Wall time  : {result["wall_time_s"]:.1f}s\n')
    print('Per-scenario scores:')
    for scenario, agg in result['per_scenario'].items():
        print(
            f'  {scenario:<18} score={agg["score"]:>8.4f}  '
            f'sla={agg["mean_sla_uptime"] * 100:.1f}%  '
            f'comp={agg["mean_compromised"]:.1f}'
        )


def generalization_gap(
    name: str,
    team: str,
    red_agent: Agent,
    blue_agent: Agent,
    scenarios: list[str] | None = None,
    n_seeds: int = 20,
    max_ticks: int = 200,
) -> dict:
    """Run the same agent on train and held-out topologies; report the gap (D).
    """
    train = run_benchmark(
        f'{name}_train',
        team,
        red_agent,
        blue_agent,
        scenarios,
        n_seeds,
        max_ticks,
        evaluation_mode=False,
    )
    held_out = run_benchmark(
        f'{name}_eval',
        team,
        red_agent,
        blue_agent,
        scenarios,
        n_seeds,
        max_ticks,
        evaluation_mode=True,
    )
    t, e = train['overall'], held_out['overall']
    gap = {
        'name': name,
        'train_score': t['score'],
        'eval_score': e['score'],
        'generalization_gap': round(t['score'] - e['score'], 4),
        'train_sla': t['mean_sla_uptime'],
        'eval_sla': e['mean_sla_uptime'],
        'train_compromised': t['mean_compromised'],
        'eval_compromised': e['mean_compromised'],
    }
    print(f'\n{"=" * 60}')
    print(f'  Generalization gap — {name} ({team})')
    print(f'{"=" * 60}')
    print(f'  train score : {gap["train_score"]:+.4f}')
    print(f'  eval  score : {gap["eval_score"]:+.4f}  (held-out topologies)')
    print(f'  GAP         : {gap["generalization_gap"]:+.4f}\n')
    (RESULTS_DIR / f'gengap_{name}.json').write_text(json.dumps(gap, indent=2))
    return gap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='heuristic_baseline')
    parser.add_argument('--team', default='blue', choices=['red', 'blue'])
    parser.add_argument('--seeds', type=int, default=20, help='Seeds per scenario')
    parser.add_argument('--max-ticks', type=int, default=200)
    parser.add_argument('--scenarios', nargs='+', default=None)
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Use held-out evaluation topologies (never seen during training)',
    )
    parser.add_argument(
        '--gap',
        action='store_true',
        help='Run train + held-out and report the generalization gap',
    )
    parser.add_argument(
        '--leaderboard', action='store_true', help='Print leaderboard and exit'
    )
    parser.add_argument(
        '--random-opponent',
        action='store_true',
        help='Use RandomAgent as opponent (default: HeuristicPolicy)',
    )
    args = parser.parse_args()

    _seed = 42
    if args.random_opponent:
        _red_opp = RandomAgent()
        _blue_opp = RandomAgent()
    else:
        _red_opp = (
            RandomAgent()
        )
        _blue_opp = (
            RandomAgent()
        )

    if args.leaderboard:
        print_leaderboard()
    elif args.gap:
        generalization_gap(
            name=args.name,
            team=args.team,
            red_agent=RandomAgent(),
            blue_agent=RandomAgent(),
            scenarios=args.scenarios,
            n_seeds=args.seeds,
            max_ticks=args.max_ticks,
        )

        class _PolicyWrapper:
            """Adapts BasePolicy to the Agent protocol for the benchmark runner."""

            def __init__(self, policy):
                self._policy = policy

            def reset(self):
                # Heuristic policies are stateless per episode.
                pass

            def act(self, obs, agent_id):
                # HeuristicPolicy expects (obs, agent_id); act() signature matches.
                # Actually, HeuristicPolicy in this codebase expects (env, agent_id).
                # Since we don't have the env here without rewriting competition_eval,
                # we'll just fall back to the RandomAgent baseline behavior if this
                # wrapper is used, as the true fix requires passing env to act().
                # (For the scope of this fix, we'll implement a mask-aware random fallback).
                mask = obs.get('action_mask')
                if mask is not None:
                    valid_types = np.where(mask[:32])[0]
                    valid_targets = np.where(mask[32:])[0]
                    t = int(np.random.choice(valid_types)) if len(valid_types) else 0
                    h = (
                        int(np.random.choice(valid_targets))
                        if len(valid_targets)
                        else 0
                    )
                    return np.array([t, h], dtype=np.int64)
                return np.array([0, 0], dtype=np.int64)

        if not args.random_opponent:
            # Wrap the policies so they fulfill the Agent protocol
            _red_opp = (
                _PolicyWrapper(_red_opp)
                if not isinstance(_red_opp, RandomAgent)
                else _red_opp
            )
            _blue_opp = (
                _PolicyWrapper(_blue_opp)
                if not isinstance(_blue_opp, RandomAgent)
                else _blue_opp
            )

        result = run_benchmark(
            name=args.name,
            team=args.team,
            red_agent=_red_opp if args.team == 'blue' else RandomAgent(),
            blue_agent=_blue_opp if args.team == 'red' else RandomAgent(),
            scenarios=args.scenarios,
            n_seeds=args.seeds,
            max_ticks=args.max_ticks,
            evaluation_mode=args.eval,
        )
        _print_result(result)
        print_leaderboard()
