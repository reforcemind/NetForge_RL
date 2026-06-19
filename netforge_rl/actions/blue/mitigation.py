from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.registry import action_registry
from netforge_rl.core.commands import (
    UpdateHostStatusCommand,
    DropSessionCommand,
    BlockPortCommand,
)


@action_registry.register('blue', 0)
class IsolateHost(BaseAction):
    """Isolates host from network."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas=[
                UpdateHostStatusCommand(self.target_ip, 'isolated'),
                DropSessionCommand(self.target_ip),
            ],
            observation_data={'alert': 'Host isolated securely.'},
        )


@action_registry.register('blue', 1)
class RestoreHost(BaseAction):
    """Re-establishes network connectivity for an isolated host."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas={
                f'hosts/{self.target_ip}/status': 'online',
                f'hosts/{self.target_ip}/privilege': 'None',
                f'hosts/{self.target_ip}/compromised_by': 'None',
            },
            observation_data={'alert': 'Host restored and cleaned.'},
        )


@action_registry.register('blue', 4)
class Remove(BaseAction):
    """Evicts unauthorized agents from a host."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas={
                f'hosts/{self.target_ip}/privilege': 'None',
                f'hosts/{self.target_ip}/compromised_by': 'None',
            },
            observation_data={'alert': 'Unauthorized access removed.'},
        )


@action_registry.register('blue', 5)
class RestoreFromBackup(BaseAction):
    """Restores host from backup."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas={
                f'hosts/{self.target_ip}/privilege': 'None',
                f'hosts/{self.target_ip}/status': 'online',
                f'hosts/{self.target_ip}/system_integrity': 'clean',
            },
            observation_data={'alert': 'Host restored from backup image.'},
        )


@action_registry.register('blue', 6)
class ConfigureACL(BaseAction):
    """Modifies firewall rules to block inbound traffic."""

    def __init__(self, agent_id: str, target_subnet: str, port: int = 445):
        super().__init__(agent_id, target_ip=target_subnet, cost=2)
        self.port = port

    def validate(self, global_state) -> bool:
        return self.target_ip in global_state.subnets

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas=[BlockPortCommand(self.target_ip, self.port)],
            observation_data={
                'alert': f'ACL configured: Drop Port {self.port} to {self.target_ip}'
            },
        )


@action_registry.register('blue', 7)
class SecurityAwarenessTraining(BaseAction):
    """Lowers human vulnerability score on subnet."""

    def __init__(self, agent_id: str, target_subnet: str):
        super().__init__(
            agent_id, target_ip=target_subnet, cost=2, financial_cost=2000, duration=3
        )

    def validate(self, global_state) -> bool:
        return self.target_ip in global_state.subnets

    def execute(self, global_state) -> ActionEffect:
        subnet = global_state.subnets.get(self.target_ip)
        if not subnet:
            return ActionEffect(success=False, state_deltas={}, observation_data={})
        deltas = {}
        for host in subnet.hosts.values():
            if hasattr(host, 'human_vulnerability_score'):
                current_score = host.human_vulnerability_score
                new_score = round(current_score * 0.2, 2)
                deltas[f'hosts/{host.ip}/human_vulnerability_score'] = new_score
        return ActionEffect(
            success=True,
            state_deltas=deltas,
            observation_data={
                'alert': f'Security Awareness Training completed on {self.target_ip}. Vulnerability drastically lowered.'
            },
            eta=self.duration,
        )
