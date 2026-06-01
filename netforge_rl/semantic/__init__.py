"""Semantic Bridge — foundation-model-native MARL wrappers.

See ROADMAP.md Phase 8 for the full design. M1 ships three primitives:

* :func:`state_to_text` — frozen :class:`EnvState` -> structured SIEM report
* :func:`build_vla_prompt` — RGB frame + report -> multimodal prompt dict
* :func:`parse_action` — model text output -> ``(action_type, target_idx)``
"""

from netforge_rl.semantic.action_menu import action_menu
from netforge_rl.semantic.la_wrapper import state_to_text
from netforge_rl.semantic.parser import parse_action
from netforge_rl.semantic.vla_wrapper import build_vla_prompt

__all__ = [
    'action_menu',
    'build_vla_prompt',
    'parse_action',
    'state_to_text',
]
