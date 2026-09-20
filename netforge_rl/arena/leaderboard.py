from __future__ import annotations

import json
from pathlib import Path


def save_leaderboard(entries: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        entries,
        key=lambda e: (
            e.get('overall', {}).get('mean', {}).get('mission_success', 0.0),
            -e.get('overall', {}).get('mean', {}).get('false_positives', 99.0),
            e.get('overall', {}).get('mean', {}).get('security', 0.0),
        ),
        reverse=True,
    )
    for i, entry in enumerate(ranked, 1):
        entry['rank'] = i
    path.write_text(json.dumps(ranked, indent=2))
    return path


def load_leaderboard(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def format_leaderboard(entries: list[dict]) -> str:
    if not entries:
        return '(empty leaderboard)'
    lines = [
        f'{"rank":<6}{"name":<22}{"split":<8}{"mission":>8}{"SLA":>8}{"FP":>8}{"CVaR":>10}'
    ]
    for e in entries:
        mean = e.get('overall', {}).get('mean', {})
        cvar = e.get('overall', {}).get('cvar05', {})
        lines.append(
            f'{e.get("rank", 0):<6}'
            f'{e.get("name", "?"):<22}'
            f'{e.get("split", "?"):<8}'
            f'{mean.get("mission_success", 0):8.2f}'
            f'{mean.get("sla_uptime", 0):8.2f}'
            f'{mean.get("false_positives", 0):8.2f}'
            f'{cvar.get("blue_return", 0):10.2f}'
        )
    return '\n'.join(lines)
