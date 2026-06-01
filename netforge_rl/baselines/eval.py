"""Baseline evaluation harness. Runs N episodes per (policy, scenario) pair
and emits results in the same JSON format as the zero-shot LLM leaderboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from netforge_rl.baselines.policies import (
    BasePolicy,
    HeuristicBluePolicy,
    HeuristicRedPolicy,
    RandomPolicy,
)
from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.semantic import EpisodeResult, append_result


POLICY_REGISTRY: dict[str, type[BasePolicy]] = {
    'random': RandomPolicy,
    'heuristic-blue': HeuristicBluePolicy,
    'heuristic-red': HeuristicRedPolicy,
}


def evaluate(
    policy: BasePolicy,
    scenario: str = 'ransomware',
    episodes: int = 5,
    max_steps: int = 100,
    seed: int = 0,
    controlled_agent: str = 'blue_dmz',
) -> list[EpisodeResult]:
    """Run ``episodes`` rollouts; the named policy controls ``controlled_agent``,
    all other agents act randomly. Returns one :class:`EpisodeResult` per episode."""
    rng = np.random.default_rng(seed)
    fallback = RandomPolicy(seed=seed)
    results: list[EpisodeResult] = []

    for ep in range(episodes):
        env = NetForgeRLEnv({'scenario_type': scenario, 'max_ticks': max_steps})
        env.reset(seed=seed + ep)

        cum_reward = {a: 0.0 for a in env.possible_agents}
        steps = 0
        while env.agents and steps < max_steps:
            actions = {}
            for agent in env.agents:
                pol = policy if agent == controlled_agent else fallback
                actions[agent] = pol.act(env, agent)
            _, rewards, term, trunc, _ = env.step(actions)
            for a, r in rewards.items():
                cum_reward[a] += float(r)
            steps += 1
            if all(term.values()) or all(trunc.values()):
                break

        compromised = sum(
            1 for h in env.global_state.all_hosts.values()
            if h.compromised_by != 'None'
        )
        isolated = sum(
            1 for h in env.global_state.all_hosts.values()
            if h.status == 'isolated'
        )
        results.append(
            EpisodeResult(
                model_id=policy.name,
                scenario=scenario,
                steps=steps,
                rewards=cum_reward,
                invalid_replies={controlled_agent: 0},
                final_compromised=compromised,
                final_isolated=isolated,
            )
        )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--policy', choices=sorted(POLICY_REGISTRY.keys()), default='heuristic-blue'
    )
    p.add_argument('--scenario', default='ransomware')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--max-steps', type=int, default=100)
    p.add_argument('--controlled-agent', default='blue_dmz')
    p.add_argument(
        '--out', type=Path, default=Path('leaderboard/baselines.json'),
    )
    args = p.parse_args()

    policy = POLICY_REGISTRY[args.policy]()
    results = evaluate(
        policy,
        scenario=args.scenario,
        episodes=args.episodes,
        max_steps=args.max_steps,
        controlled_agent=args.controlled_agent,
    )
    for r in results:
        append_result(args.out, r)
    print(f'Wrote {len(results)} episodes to {args.out}')


if __name__ == '__main__':
    main()
