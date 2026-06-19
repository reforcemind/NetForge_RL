import random

from netforge_rl.semantic.action_menu import action_menu


class MockLLMClient:
    """Drives a deterministic policy without API keys."""

    model_id = 'mock'

    def __init__(self, replies=None, seed=0):
        self._replies = list(replies) if replies else None
        self._i = 0
        self._rng = random.Random(seed)

    def act(self, prompt):
        if self._replies is not None:
            r = self._replies[self._i % len(self._replies)]
            self._i += 1
            return r

        agent_id = prompt.get('_agent_id', 'blue_dmz')
        ips = prompt.get('_target_ips') or ['10.0.0.1']
        menu = action_menu(agent_id)
        action_id = self._rng.choice(list(menu.keys())) if menu else 0
        target = self._rng.choice(ips)
        return f'ACTION {action_id} TARGET {target}'
