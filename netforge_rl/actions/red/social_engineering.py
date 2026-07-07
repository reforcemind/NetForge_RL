from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.commands import (
    EstablishSessionCommand,
    UpdateHostPrivilegeCommand,
)
from netforge_rl.core.registry import action_registry


@action_registry.register('red', 21)
class SpearPhishing(BaseAction):
    """Executes spear-phishing."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(
            agent_id,
            target_ip=target_ip,
            cost=2,
            financial_cost=50,
            duration=15,
            required_prior_state=None,
        )

    def validate(self, global_state) -> bool:
        if not self.target_ip or self.target_ip not in global_state.all_hosts:
            return False
        host = global_state.all_hosts[self.target_ip]
        if 'Windows' not in getattr(host, 'os', ''):
            return False
        return True

    def execute(self, global_state) -> ActionEffect:
        host = global_state.all_hosts.get(self.target_ip)
        phish_chance = getattr(host, 'human_vulnerability_score', 0.1)
        if global_state.rng.random() > phish_chance:
            return ActionEffect(
                success=False,
                state_deltas=[],
                observation_data={
                    'phishing': 'failed',
                    'reason': 'user reported suspicious email',
                },
            )
        deltas = [
            UpdateHostPrivilegeCommand(
                self.target_ip, 'User', compromised_by=self.agent_id
            ),
            EstablishSessionCommand(self.agent_id, self.target_ip, port=443),
        ]
        return ActionEffect(
            success=True,
            state_deltas=deltas,
            observation_data={
                'phishing': 'success',
                'status': 'C2 Session Established via user execution',
            },
            eta=self.duration,
        )
