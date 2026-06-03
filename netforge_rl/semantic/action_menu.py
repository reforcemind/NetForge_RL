import netforge_rl.actions  # noqa: F401  populate the registry
from netforge_rl.core.registry import action_registry


def _team_keys_for(agent_id):
    lower = agent_id.lower()
    if 'red' in lower:
        primary = 'red_commander' if 'commander' in lower else 'red'
    else:
        primary = 'blue_commander' if 'commander' in lower else 'blue'
    return [primary, lower]


def action_menu(agent_id):
    """Return ``{action_type_id: ActionClassName}`` for an agent."""
    menu = {}
    for team in _team_keys_for(agent_id):
        for gid, cls in action_registry._actions.get(team, {}).items():
            menu.setdefault(gid, cls.__name__)
    return dict(sorted(menu.items()))
