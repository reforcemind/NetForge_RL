import netforge_rl.actions  # noqa: F401
from netforge_rl.core.registry import action_registry, team_of


def action_menu(agent_id):
    """Return ``{action_type_id: ActionClassName}`` for an agent."""
    menu = {
        gid: cls.__name__
        for gid, cls in action_registry._actions.get(team_of(agent_id), {}).items()
    }
    return dict(sorted(menu.items()))
