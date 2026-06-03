import re

from netforge_rl.semantic.action_menu import action_menu


_ACTION_RE = re.compile(
    r'ACTION\s+(\d+)\s+TARGET\s+([\w\.\:]+)',
    re.IGNORECASE,
)


def parse_action(text, agent_id, target_ips):
    """Parse `ACTION <id> TARGET <ip>` -> (action_type_id, target_idx) or None."""
    match = _ACTION_RE.search(text)
    if not match:
        return None
    action_id = int(match.group(1))
    target = match.group(2)

    if action_id not in action_menu(agent_id):
        return None
    if target not in target_ips:
        return None

    return action_id, target_ips.index(target)
