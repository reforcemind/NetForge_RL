from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from netforge_rl.diagnostics.base import all_diagnostics, run_diagnostic


def capability_card(
    policy_factory,
    seeds: Sequence[int] = (0, 1, 2),
    out_dir: Optional[str] = None,
    name: str = 'policy',
) -> dict:
    """Score a policy on every diagnostic capability, averaged over seeds.
    """
    per_capability: dict[str, list[float]] = {}
    for seed in seeds:
        for probe in all_diagnostics():
            result = run_diagnostic(probe, policy_factory(), seed=seed)
            per_capability.setdefault(result.capability, []).append(result.score)

    capabilities = {
        cap: round(float(np.mean(scores)), 4)
        for cap, scores in sorted(per_capability.items())
    }
    card = {
        'name': name,
        'seeds': list(seeds),
        'capabilities': capabilities,
        'capability_std': {
            cap: round(float(np.std(scores)), 4)
            for cap, scores in sorted(per_capability.items())
        },
        'overall': round(float(np.mean(list(capabilities.values()))), 4),
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f'{name}_card.json').write_text(json.dumps(card, indent=2))
        _render_radar(card, out / f'{name}_card.png')
    return card


def _render_radar(card: dict, path: Path) -> None:
    """Render a radar chart of the capability scores. No-op if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return

    caps = list(card['capabilities'])
    values = [card['capabilities'][c] for c in caps]
    angles = np.linspace(0, 2 * np.pi, len(caps), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'polar': True})
    ax.plot(angles, values, color='#2a9d8f', linewidth=2)
    ax.fill(angles, values, color='#2a9d8f', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(caps)
    ax.set_ylim(0, 1)
    ax.set_title(f'{card["name"]} — capability card (overall {card["overall"]:.2f})')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
