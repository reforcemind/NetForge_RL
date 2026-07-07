import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from netforge_rl.baselines import (
    HeuristicBluePolicy,
    HeuristicRedPolicy,
    KillChainRedPolicy,
    RandomPolicy,
    evaluate,
)
from netforge_rl.semantic import summarize


POLICIES = {
    'random': RandomPolicy,
    'heuristic-blue': HeuristicBluePolicy,
    'heuristic-red': HeuristicRedPolicy,
    'killchain-red': KillChainRedPolicy,
}

SCENARIOS = ('ransomware', 'apt_espionage', 'iot_grid', 'ot_stuxnet')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--max-steps', type=int, default=80)
    p.add_argument('--controlled-agent', default='blue_dmz')
    p.add_argument(
        '--results',
        type=Path,
        default=Path('leaderboard/baselines.json'),
    )
    p.add_argument(
        '--summary',
        type=Path,
        default=Path('leaderboard/baselines_summary.json'),
    )
    args = p.parse_args()

    args.results.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.perf_counter()
    for policy_name, policy_cls in POLICIES.items():
        for scen in SCENARIOS:
            controlled = (
                'red_operator'
                if policy_name in ('heuristic-red', 'killchain-red')
                else args.controlled_agent
            )
            for r in evaluate(
                policy_cls(seed=0),
                scenario=scen,
                episodes=args.episodes,
                max_steps=args.max_steps,
                controlled_agent=controlled,
            ):
                rows.append(asdict(r))

    args.results.write_text(json.dumps(rows, indent=2))
    summary = summarize(args.results)
    args.summary.write_text(json.dumps(summary, indent=2))

    wall = time.perf_counter() - t0
    print(
        f'Ran {len(rows)} episodes across {len(POLICIES) * len(SCENARIOS)} cells '
        f'in {wall:.1f}s'
    )
    print(f'Results: {args.results}')
    print(f'Summary: {args.summary}\n')
    print(
        f'{"model":<16} {"scenario":<16} {"n":>3} {"reward":>10} {"comp":>6} {"iso":>5}'
    )
    for row in summary:
        print(
            f'{row["model_id"]:<16} {row["scenario"]:<16} '
            f'{row["n_episodes"]:>3} {row["mean_total_reward"]:>10.2f} '
            f'{row["mean_compromised"]:>6.1f} {row["mean_isolated"]:>5.1f}'
        )


if __name__ == '__main__':
    main()
