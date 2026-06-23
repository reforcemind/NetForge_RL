from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from netforge_rl.core.state import GlobalNetworkState
    from netforge_rl.core.action import ActionEffect


class BaseScenario(ABC):
    """Abstract Scenario outlining reward dynamics and target objectives."""

    MAX_STEP_REWARD: float = 10.0

    @abstractmethod
    def calculate_reward(
        self,
        agent_id: str,
        global_state: 'GlobalNetworkState',
        effect: 'ActionEffect' = None,
    ) -> float:
        pass

    @abstractmethod
    def check_termination(self, global_state: 'GlobalNetworkState') -> Dict[str, bool]:
        pass

    def normalized_reward(self, reward: float) -> float:
        """Reward divided by MAX_STEP_REWARD for cross-scenario comparability."""
        return reward / max(self.MAX_STEP_REWARD, 1e-6)
