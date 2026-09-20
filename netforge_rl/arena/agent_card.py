from __future__ import annotations

from collections.abc import Sequence

from netforge_rl.arena.evaluate import evaluate_split
from netforge_rl.arena.spec import ARENA_2026
from netforge_rl.diagnostics.capability_card import capability_card


def build_agent_card(
    policy_factory,
    *,
    name: str = 'policy',
    seeds: Sequence[int] = (0, 1),
    opponent_factory=None,
    run_arena: bool = True,
    out_dir: str | None = None,
) -> dict:
    """Capability probes plus optional Arena-dev metrics."""
    card = capability_card(policy_factory, seeds=seeds, out_dir=out_dir, name=name)
    caps = card['capabilities']
    fpr = round(max(0.0, 1.0 - float(caps.get('precision', 1.0))), 4)
    headlines = {
        'Safety': caps.get('safety', 0.0),
        'OOD Generalization': caps.get('generalization', 0.0),
        'Memory': caps.get('memory', 0.0),
        'False Positive Rate': fpr,
        'Temporal': caps.get('temporal', 0.0),
        'Deception Resistance': caps.get('deception', 0.0),
        'Adaptation': caps.get('adaptation', 0.0),
        'Attention': caps.get('attention', 0.0),
    }
    card['headlines'] = headlines
    card['false_positive_rate'] = fpr

    if run_arena:
        from netforge_rl.baselines.policies import KillChainRedPolicy

        red_factory = opponent_factory or (lambda: KillChainRedPolicy(seed=0))
        arena = evaluate_split(
            red_factory,
            policy_factory,
            split='dev',
            scenarios=('ransomware',),
            seeds=tuple(seeds)[:2],
            max_ticks=40,
            spec=ARENA_2026,
        )
        card['arena_dev'] = arena['overall']
        mean = arena['overall'].get('mean', {})
        card['headlines']['Mission Success'] = mean.get('mission_success', 0.0)
        card['headlines']['CVaR Return'] = (
            arena['overall'].get('cvar05', {}).get('blue_return', 0.0)
        )
    if out_dir:
        import json
        from pathlib import Path

        path = Path(out_dir) / f'{name}_card.json'
        path.write_text(json.dumps(card, indent=2))
    return card


def format_agent_card(card: dict) -> str:
    lines = [f'Agent Card: {card.get("name", "policy")}']
    overall = card.get('overall')
    if overall is not None:
        lines.append(f'  Overall              {overall:.2f}')
    for label, value in card.get('headlines', {}).items():
        lines.append(f'  {label:<22} {float(value):.2f}')
    return '\n'.join(lines)
