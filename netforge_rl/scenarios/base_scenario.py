import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

from netforge_rl.core.commands import iter_host_deltas

if TYPE_CHECKING:
    from netforge_rl.core.state import GlobalNetworkState
    from netforge_rl.core.action import ActionEffect

PADDING_SUBNET = '169.254.0.0/16'


class BaseScenario(ABC):
    """Reward dynamics and termination for a red/blue scenario.

    Subclasses implement `_red_reward`, `_blue_reward`, and `_is_terminal`; the shared
    action-cost penalty, red/blue dispatch, and per-agent termination dict are handled
    here.
    """

    MAX_STEP_REWARD: float = 10.0
    ACTION_COST_FACTOR: float = 0.05
    REWARD_WEIGHTS: Dict[str, Dict[str, float]] = {
        'red_weights': {},
        'blue_weights': {},
    }

    def __init__(self, agents):
        self.agents = agents

    def calculate_reward(
        self,
        agent_id: str,
        global_state: 'GlobalNetworkState',
        effect: 'ActionEffect' = None,
    ) -> float:
        reward = 0.0
        if effect and getattr(effect, 'cost', 0) > 0:
            reward -= effect.cost * self.ACTION_COST_FACTOR
        if 'red' in agent_id.lower():
            reward += self._red_reward(agent_id, global_state, effect)
        else:
            reward += self._blue_reward(agent_id, global_state, effect)
        return reward

    def check_termination(self, global_state: 'GlobalNetworkState') -> Dict[str, bool]:
        done = self._is_terminal(global_state)
        return {agent: done for agent in self.agents}

    def normalized_reward(self, reward: float) -> float:
        """Reward mapped into [-1, 1]."""
        return math.tanh(reward / max(self.MAX_STEP_REWARD, 1e-6))

    @abstractmethod
    def _red_reward(self, agent_id, state, effect) -> float: ...

    @abstractmethod
    def _blue_reward(self, agent_id, state, effect) -> float: ...

    @abstractmethod
    def _is_terminal(self, state) -> bool: ...

    @staticmethod
    def _iter_deltas(effect):
        """Yield (attr, ip, val) host deltas from a successful effect; nothing otherwise."""
        if effect and effect.success and effect.state_deltas:
            yield from iter_host_deltas(effect.state_deltas)

    @staticmethod
    def _hosts_in_subnet(state, name: str) -> list:
        return [
            h
            for h in state.all_hosts.values()
            if state.get_subnet_name(h.subnet_cidr) == name
        ]

    @staticmethod
    def _any_kinetic(state) -> bool:
        return any(
            getattr(h, 'system_integrity', 'clean') == 'kinetic_destruction'
            for h in state.all_hosts.values()
        )

    @staticmethod
    def _failed_penalty(effect, penalty: float = -0.1) -> float:
        return penalty if (effect and not effect.success) else 0.0
