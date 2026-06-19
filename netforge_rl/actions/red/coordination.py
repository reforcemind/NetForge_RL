from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.registry import action_registry

@action_registry.register('red_commander', 3)
class ShareIntelligence(BaseAction):
    """Shares discovered hosts with allied agent."""

    def __init__(self, agent_id: str, target_agent_id: str):
        super().__init__(agent_id, target_ip=target_agent_id, cost=1)
        self.target_agent_id = target_agent_id

    def validate(self, global_state) -> bool:
        if self.agent_id not in global_state.agent_knowledge:
            return False
        return True

    def execute(self, global_state) -> ActionEffect:
        knowledge_deltas = {}
        if self.agent_id in global_state.agent_knowledge:
            known_ips = global_state.agent_knowledge[self.agent_id]
            for known_ip in known_ips:
                knowledge_deltas[f'knowledge/{self.target_agent_id}/{known_ip}'] = 'True'
        return ActionEffect(success=True, state_deltas=knowledge_deltas, observation_data={'shared_intel_with': self.target_agent_id})