"""Per-agent legal action menu built from the action registry."""

from __future__ import annotations

import netforge_rl.actions  # noqa: F401  populate the registry
from netforge_rl.core.registry import action_registry


def _team_keys_for(agent_id: str) -> list[str]:
    """Mirrors the registry's lookup order in ``get_action_class``."""
    lower = agent_id.lower()
    if 'red' in lower:
        primary = 'red_commander' if 'commander' in lower else 'red'
    else:
        primary = 'blue_commander' if 'commander' in lower else 'blue'
    return [primary, lower]


def action_menu(agent_id: str) -> dict[int, str]:
    """Return ``{action_type_id: ActionClassName}`` for an agent."""
    menu: dict[int, str] = {}
    for team in _team_keys_for(agent_id):
        for gid, cls in action_registry._actions.get(team, {}).items():
            menu.setdefault(gid, cls.__name__)
    return dict(sorted(menu.items()))
