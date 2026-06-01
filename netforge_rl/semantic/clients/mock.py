"""Mock LLM client for tests + harness smoke runs.

Drives a deterministic policy without needing API keys. Two modes:

* ``script``  — return a fixed list of replies in order, cycling.
* ``random``  — emit a uniformly-random legal ACTION line. Requires the
  caller to inject ``agent_id`` and ``target_ips`` into the prompt dict
  under the keys ``_agent_id`` / ``_target_ips`` (the harness does this).
"""

from __future__ import annotations

import random
from typing import Any

from netforge_rl.semantic.action_menu import action_menu


class MockLLMClient:
    model_id = 'mock'

    def __init__(self, replies: list[str] | None = None, seed: int = 0):
        self._replies = list(replies) if replies else None
        self._i = 0
        self._rng = random.Random(seed)

    def act(self, prompt: dict[str, Any]) -> str:
        if self._replies is not None:
            r = self._replies[self._i % len(self._replies)]
            self._i += 1
            return r

        # Random mode — pull metadata the harness attaches.
        agent_id = prompt.get('_agent_id', 'blue_dmz')
        ips = prompt.get('_target_ips') or ['10.0.0.1']
        menu = action_menu(agent_id)
        action_id = self._rng.choice(list(menu.keys())) if menu else 0
        target = self._rng.choice(ips)
        return f'ACTION {action_id} TARGET {target}'
