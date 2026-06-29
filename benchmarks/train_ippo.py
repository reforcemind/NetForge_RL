from __future__ import annotations

import argparse
import json
from pathlib import Path

from netforge_rl.backends.jax.vector_env import SCENARIO_IDS
from netforge_rl.baselines.jax_ppo import PPOConfig, ippo_train

RESULTS_DIR = Path(__file__).parent / 'results'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', default='ransomware', choices=list(SCENARIO_IDS))
    p.add_argument('--iters', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-steps', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    cfg = PPOConfig(
        total_iters=args.iters,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        n_blue=3,
        scenario=SCENARIO_IDS[args.scenario],
    )
    print(
        f'Training IPPO on {args.scenario} for {args.iters} iters '
        f'(batch={args.batch_size}, steps={args.num_steps})...'
    )
    out = ippo_train(cfg)

    curve = out['reward_curve']
    w = max(1, len(curve) // 10)
    start = sum(curve[:w]) / w
    end = sum(curve[-w:]) / w
    improvement = end - start

    print(f'\n  mean blue reward  first {w} iters: {start:+.4f}')
    print(f'  mean blue reward  last  {w} iters: {end:+.4f}')
    print(
        f'  improvement                      : {improvement:+.4f}  '
        f'({"learned" if improvement > 0 else "no gain"})'
    )

    result = {
        'scenario': args.scenario,
        'iters': args.iters,
        'config': vars(args),
        'reward_curve': curve,
        'loss_curve': out['losses'],
        'reward_start': round(start, 4),
        'reward_end': round(end, 4),
        'improvement': round(improvement, 4),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f'ippo_{args.scenario}.json'
    out_path.write_text(json.dumps(result, indent=2))
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
