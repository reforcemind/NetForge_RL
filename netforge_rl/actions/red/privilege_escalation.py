from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.registry import action_registry


@action_registry.register('red', 1)
class PrivilegeEscalate(BaseAction):
    """Executes a local privilege escalation exploit."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        host = global_state.all_hosts.get(self.target_ip)
        if not host or host.privilege != 'User':
            return False
        return global_state.can_route_to(self.target_ip, agent_id=self.agent_id)

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas={
                f'hosts/{self.target_ip}/privilege': 'Root',
                f'hosts/{self.target_ip}/compromised_by': self.agent_id,
            },
            observation_data={'privilege': 'escalated'},
        )


@action_registry.register('red', 6)
class JuicyPotato(BaseAction):
    """Executes the JuicyPotato privilege escalation vector."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        host = global_state.all_hosts.get(self.target_ip)
        if not host or host.privilege != 'User':
            return False
        if 'Windows' not in host.os:
            return False
        return global_state.can_route_to(self.target_ip, agent_id=self.agent_id)

    def execute(self, global_state) -> ActionEffect:
        host = global_state.all_hosts.get(self.target_ip)
        if host and 'Windows' not in host.os:
            return ActionEffect(
                success=False,
                state_deltas={},
                observation_data={'privilege': 'failed - OS is not Windows'},
            )
        return ActionEffect(
            success=True,
            state_deltas={
                f'hosts/{self.target_ip}/privilege': 'Root',
                f'hosts/{self.target_ip}/compromised_by': self.agent_id,
            },
            observation_data={'privilege': 'JuicyPotato elevated'},
        )


@action_registry.register('red', 10)
class V4L2KernelExploit(BaseAction):
    """Executes a V4L2 kernel-level vulnerability."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        host = global_state.all_hosts.get(self.target_ip)
        if not host or host.privilege != 'User':
            return False
        if 'Linux' not in host.os:
            return False
        return global_state.can_route_to(self.target_ip, agent_id=self.agent_id)

    def execute(self, global_state) -> ActionEffect:
        host = global_state.all_hosts.get(self.target_ip)
        if host and ('Linux' not in host.os or 'V4L2' not in host.vulnerabilities):
            return ActionEffect(
                success=False,
                state_deltas={},
                observation_data={
                    'privilege': 'failed - target patched or incompatible OS'
                },
            )
        return ActionEffect(
            success=True,
            state_deltas={
                f'hosts/{self.target_ip}/privilege': 'Root',
                f'hosts/{self.target_ip}/compromised_by': self.agent_id,
            },
            observation_data={'privilege': 'V4L2 Kernel escalated'},
        )


@action_registry.register('red_operator', 9)
class PassTheHash(BaseAction):
    """Moves laterally using stolen Kerberos or NTLM hashes."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip, cost=1)

    def validate(self, global_state) -> bool:
        has_dc_hash = False
        for host in global_state.all_hosts.values():
            if host.is_domain_controller and host.privilege == 'Root':
                if host.compromised_by == self.agent_id:
                    has_dc_hash = True
                    break
        if not has_dc_hash:
            return False
        return global_state.can_route_to(self.target_ip, agent_id=self.agent_id)

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas={
                f'hosts/{self.target_ip}/privilege': 'Root',
                f'hosts/{self.target_ip}/compromised_by': self.agent_id,
            },
            observation_data={'privilege': 'Pass-The-Hash lateral pivot successful.'},
        )
