from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from netforge_rl.baselines.jax_ppo import PPOConfig, ippo_train, save_params

RESULTS_DIR = Path(__file__).parent / 'results'
SCENARIOS = ['ransomware', 'apt_espionage', 'cloud_hybrid', 'iot_grid', 'ot_stuxnet']


def train_and_record(cfg: PPOConfig, name: str) -> dict:
    """Train IPPO on the JAX backend and record the learning curve + checkpoint."""
    t0 = time.perf_counter()
    out = ippo_train(cfg)
    wall = time.perf_counter() - t0

    rc = out['reward_curve']
    n = max(len(rc) // 5, 1)
    record = {
        'name': name,
        'scenario': SCENARIOS[cfg.scenario],
        'iterations': cfg.total_iters,
        'num_steps': cfg.num_steps,
        'batch_size': cfg.batch_size,
        'env_steps': cfg.total_iters * cfg.num_steps * cfg.batch_size,
        'wall_time_s': round(wall, 1),
        'reward_curve': [round(float(x), 5) for x in rc],
        'loss_curve': [round(float(x), 5) for x in out['losses']],
        'reward_first_window': round(float(sum(rc[:n]) / n), 5),
        'reward_last_window': round(float(sum(rc[-n:]) / n), 5),
    }
    record['improvement'] = round(
        record['reward_last_window'] - record['reward_first_window'], 5
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f'ippo_curve_{name}.json').write_text(json.dumps(record, indent=2))
    save_params(out['params'], str(RESULTS_DIR / f'ippo_{name}.npz'))
    _render_curve(record, RESULTS_DIR / f'ippo_curve_{name}.png')
    return record


def _render_curve(record: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(record['reward_curve'], color='#2a9d8f', linewidth=2)
    ax.set_xlabel('PPO iteration')
    ax.set_ylabel('mean blue reward / step')
    ax.set_title(
        f'IPPO on {record["scenario"]} (+{record["improvement"]:.3f} over training)'
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', default='blue')
    p.add_argument('--iters', type=int, default=40)
    p.add_argument('--num-steps', type=int, default=48)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--scenario', type=int, default=0, help='index into SCENARIOS')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    cfg = PPOConfig(
        total_iters=args.iters,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        num_minibatches=4,
        scenario=args.scenario,
        seed=args.seed,
    )
    rec = train_and_record(cfg, args.name)
    print(
        f'{rec["name"]}: {rec["scenario"]}  '
        f'{rec["reward_first_window"]:.3f} -> {rec["reward_last_window"]:.3f} '
        f'(+{rec["improvement"]:.3f}) over {rec["env_steps"]:,} env-steps '
        f'in {rec["wall_time_s"]}s'
    )
