from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.registry import action_registry


@action_registry.register('blue', 8)
class DeployEDR(BaseAction):
    """Installs endpoint detection on a host, enabling Analyze to reveal IoCs."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(
            agent_id, target_ip=target_ip, cost=3, financial_cost=200, duration=2
        )

    def validate(self, global_state) -> bool:
        host = global_state.all_hosts.get(self.target_ip)
        return host is not None and host.status == 'online'

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas={f'hosts/{self.target_ip}/edr_active': True},
            observation_data={'edr_deployed': self.target_ip},
            eta=self.duration,
        )
