"""Parse a model's text reply into an env action ID.

The protocol the LA wrapper instructs the model to use is:

    ACTION <action_id> TARGET <host_ip>

This module accepts that form (case-insensitive, free-form whitespace,
optional ``#`` comments, optional preamble) and returns
``(action_type_id, target_idx)`` ready for ``env.step``. Hallucinated
action_ids or unknown IPs return ``None`` rather than raising — the
caller chooses how to handle invalid replies (resample, no-op, penalty).
"""

from __future__ import annotations

import re

from netforge_rl.semantic.action_menu import action_menu


_ACTION_RE = re.compile(
    r'ACTION\s+(\d+)\s+TARGET\s+([\w\.\:]+)',
    re.IGNORECASE,
)


def parse_action(
    text: str,
    agent_id: str,
    target_ips: list[str],
) -> tuple[int, int] | None:
    """Return ``(action_type_id, target_idx)`` or ``None`` if unparseable."""
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
