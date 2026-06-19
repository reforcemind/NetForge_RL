from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from netforge_rl.core.state import GlobalNetworkState
    from netforge_rl.core.action import ActionEffect


class BaseScenario(ABC):
    """Abstract Scenario outlining the Reward dynamics and Target objectives."""

    @abstractmethod
    def calculate_reward(
        self,
        agent_id: str,
        global_state: 'GlobalNetworkState',
        effect: 'ActionEffect' = None,
    ) -> float:
        """Dynamically calculates the reward for the specified agent."""
        pass

    @abstractmethod
    def check_termination(self, global_state: 'GlobalNetworkState') -> Dict[str, bool]:
        """Returns a dictionary mapping agent_ids to their termination state."""
        pass
