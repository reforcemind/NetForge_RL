import json
from dataclasses import asdict
from pathlib import Path

from netforge_rl.semantic.runner import EpisodeResult


def append_result(path, result: EpisodeResult):
    """Append a single episode result to ``path`` (creates if missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        existing = json.loads(path.read_text() or '[]')
    existing.append(asdict(result))
    path.write_text(json.dumps(existing, indent=2))
    return path


def summarize(path):
    """Return per-(model, scenario) aggregates ordered by mean total reward."""
    rows = json.loads(Path(path).read_text())
    groups = {}
    for r in rows:
        groups.setdefault((r['model_id'], r['scenario']), []).append(r)

    summary = []
    for (model, scen), rs in groups.items():
        total_rewards = [sum(r['rewards'].values()) for r in rs]
        summary.append(
            {
                'model_id': model,
                'scenario': scen,
                'n_episodes': len(rs),
                'mean_total_reward': sum(total_rewards) / len(rs),
                'mean_compromised': sum(r['final_compromised'] for r in rs) / len(rs),
                'mean_isolated': sum(r['final_isolated'] for r in rs) / len(rs),
                'mean_invalid_replies': sum(
                    sum(r['invalid_replies'].values()) for r in rs
                )
                / len(rs),
            }
        )
    summary.sort(key=lambda r: r['mean_total_reward'], reverse=True)
    return summary
