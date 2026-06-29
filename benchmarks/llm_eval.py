from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.semantic.clients.mock import MockLLMClient
from netforge_rl.semantic.modes import zero_shot_attacker, zero_shot_defender

RESULTS_DIR = Path(__file__).parent / 'results'


def build_client(backend, model):
    """Return the LLM client to evaluate."""
    if backend == 'vllm':
        from netforge_rl.semantic.clients.vllm_client import VLLMClient

        return VLLMClient(model=model)
    return MockLLMClient(seed=0)


def evaluate(mode, scenario, n_seeds, max_steps, backend='mock', model=''):
    client = build_client(backend, model)
    side = 'blue' if mode == 'defender' else 'red'
    runner = zero_shot_defender if mode == 'defender' else zero_shot_attacker

    per_seed = []
    for seed in range(n_seeds):
        env = NetForgeRLEnv({'scenario_type': scenario, 'max_ticks': max_steps})
        out = runner(env, client, max_steps=max_steps, seed=seed)
        controlled = [a for a in out['cum_reward'] if side in a]
        per_seed.append(
            {
                'seed': seed,
                'reward': float(sum(out['cum_reward'][a] for a in controlled)),
                'invalid_replies': int(
                    sum(out['invalid_replies'][a] for a in controlled)
                ),
                'final_compromised': out['final_compromised'],
                'final_isolated': out['final_isolated'],
                'steps': out['steps'],
            }
        )

    rewards = np.array([r['reward'] for r in per_seed], dtype=float)
    n = len(rewards)
    ci95 = 1.96 * float(rewards.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    summary = {
        'mode': mode,
        'model_id': client.model_id,
        'scenario': scenario,
        'n_seeds': n,
        'mean_reward': round(float(rewards.mean()), 4),
        'ci95_reward': round(ci95, 4),
        'total_invalid_replies': int(sum(r['invalid_replies'] for r in per_seed)),
        'per_seed': per_seed,
    }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['defender', 'attacker'], default='defender')
    p.add_argument('--scenario', default='ransomware')
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--max-steps', type=int, default=50)
    p.add_argument('--backend', choices=['mock', 'vllm'], default='mock')
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    args = p.parse_args()

    summary = evaluate(
        args.mode, args.scenario, args.seeds, args.max_steps, args.backend, args.model
    )
    print(
        f'LLM {args.mode} [{summary["model_id"]}] on {args.scenario}: '
        f'reward {summary["mean_reward"]:+.3f} +/- {summary["ci95_reward"]:.3f}  '
        f'({summary["total_invalid_replies"]} invalid replies over '
        f'{summary["n_seeds"]} seeds)'
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f'llm_{args.mode}_{args.scenario}.json').write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == '__main__':
    main()
