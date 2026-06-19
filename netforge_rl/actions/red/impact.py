from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.registry import action_registry
from netforge_rl.core.commands import ConsumeBandwidthCommand

@action_registry.register('red_operator', 2)
class Impact(BaseAction):
    """Executes ransomware/wiper impact."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(success=True, state_deltas={f'hosts/{self.target_ip}/system_integrity': 'compromised'}, observation_data={'impact': 'executed'})

@action_registry.register('red_operator', 8)
class KillProcess(BaseAction):
    """Terminates specific process."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip, cost=1)

    def validate(self, global_state) -> bool:
        return global_state.can_route_to(self.target_ip, agent_id=self.agent_id)

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(success=True, state_deltas={f'hosts/{self.target_ip}/edr_active': False}, observation_data={'kill_process': 'EDR blinded'})

@action_registry.register('red_operator', 10)
class ExfiltrateData(BaseAction):
    """Exfiltrates data."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip, duration=3)

    def validate(self, global_state) -> bool:
        host = global_state.all_hosts.get(self.target_ip)
        if not host or host.privilege not in ['User', 'Root']:
            return False
        return global_state.can_route_to(self.target_ip, agent_id=self.agent_id)

    def execute(self, global_state) -> ActionEffect:
        host = global_state.all_hosts.get(self.target_ip)
        target_subnet = host.subnet_cidr if host else 'unknown'
        deltas = [ConsumeBandwidthCommand(target_subnet, amount=500)]
        return ActionEffect(success=True, state_deltas=deltas, observation_data={'action': 'exfiltrated_data_chunk'})