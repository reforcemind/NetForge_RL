from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.competition_eval import Agent, RandomAgent
from benchmarks.run_benchmark import ALL_SCENARIOS, run_benchmark

RESULTS_DIR = Path(__file__).parent / 'results'


class MaskAwareGreedyBlue:
    def reset(self) -> None:
        pass

    def act(self, obs: dict, agent_id: str) -> np.ndarray:
        mask = obs.get('action_mask')
        if mask is None:
            return np.array([0, 0], dtype=np.int64)
        live_targets = np.where(mask[32:])[0]
        target = int(live_targets[0]) if len(live_targets) else 0
        return np.array([0, target], dtype=np.int64)


AGENTS: dict[str, type[Agent]] = {
    'random': RandomAgent,
    'greedy-blue': MaskAwareGreedyBlue,
}


def sweep(agent_names, team, scenarios, n_seeds, max_ticks) -> list[dict]:
    rows = []
    for name in agent_names:
        agent = AGENTS[name]()
        result = run_benchmark(
            name=f'sweep-{name}',
            team=team,
            red_agent=RandomAgent(),
            blue_agent=agent,
            scenarios=scenarios,
            n_seeds=n_seeds,
            max_ticks=max_ticks,
        )
        o = result['overall']
        rows.append(
            {
                'agent': name,
                'score': o['score'],
                'sla_uptime': o['mean_sla_uptime'],
                'ci95_sla': o['ci95_sla_uptime'],
                'compromised': o['mean_compromised'],
                'ci95_comp': o['ci95_compromised'],
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--agents', nargs='+', default=list(AGENTS.keys()))
    p.add_argument('--team', default='blue', choices=['red', 'blue'])
    p.add_argument('--scenarios', nargs='+', default=ALL_SCENARIOS)
    p.add_argument('--seeds', type=int, default=20)
    p.add_argument('--max-ticks', type=int, default=200)
    args = p.parse_args()

    rows = sweep(args.agents, args.team, args.scenarios, args.seeds, args.max_ticks)

    print(f'\n{"=" * 64}')
    print(
        f'  Baseline sweep ({args.team}) — {args.seeds} seeds x '
        f'{len(args.scenarios)} scenarios'
    )
    print(f'{"=" * 64}')
    print(f'  {"agent":<16}{"score":>10}{"SLA%":>9}{"comp":>8}')
    for r in sorted(rows, key=lambda x: x['score'], reverse=True):
        print(
            f'  {r["agent"]:<16}{r["score"]:>10.3f}'
            f'{r["sla_uptime"] * 100:>8.1f}%{r["compromised"]:>8.1f}'
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'baseline_sweep.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
