from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.registry import action_registry
from netforge_rl.core.commands import UpdateHostPrivilegeCommand


@action_registry.register('red', 1)
class PrivilegeEscalate(BaseAction):
    """Executes a generic local privilege escalation exploit on a compromised

    host.

    Elevates an agent's access from standard 'User' to 'Root' or 'SYSTEM', granting
    unrestricted control over the endpoint for subsequent impact actions.

    Args:
        agent_id (str): The unique identifier of the Red agent.
        target_ip (str): The IP address of the already compromised host.
    """

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


@action_registry.register('red_operator', 6)
class JuicyPotato(BaseAction):
    """Simulates the JuicyPotato local privilege escalation vector leveraging

    DCOM on Windows.

    Abuses `SeImpersonatePrivilege` to elevate a service account to `NT AUTHORITY\\SYSTEM`.

    Args:
        agent_id (str): Reference to the executing Red operator.
        target_ip (str): Target IPv4 string.
    """

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


@action_registry.register('red_operator', 7)
class V4L2KernelExploit(BaseAction):
    """Executes a specific kernel-level vulnerability via Video4Linux (V4L2) on

    Linux targets.

    Targets memory corruption in outdated kernel modules to spawn a root shell.

    Args:
        agent_id (str): Reference to the executing Red operator.
        target_ip (str): Target IPv4 string.
    """

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
    """Executes a lateral movement attack bypassing authentication using
    Kerberos / NTLM hashes extracted from a Domain Controller.

    Args:
        agent_id (str): Reference to the executing Red operator.
        target_ip (str): Target IPv4 string (can be un-exploited if DC is cracked).
    """

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip, cost=1)

    def validate(self, global_state) -> bool:
        has_dc_hash = False
        for host in global_state.all_hosts.values():
            if host.is_domain_controller and host.privilege in ['Root', 'SYSTEM']:
                if host.compromised_by == self.agent_id:
                    has_dc_hash = True
                    break

        if not has_dc_hash:
            return False

        return global_state.can_route_to(self.target_ip, agent_id=self.agent_id)

    def execute(self, global_state) -> ActionEffect:

        deltas = [
            UpdateHostPrivilegeCommand(
                self.target_ip, 'Root', compromised_by=self.agent_id
            )
        ]

        return ActionEffect(
            success=True,
            state_deltas=deltas,
            observation_data={'privilege': 'Pass-The-Hash lateral pivot successful.'},
        )
