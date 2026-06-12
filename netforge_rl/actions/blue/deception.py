from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.registry import action_registry


@action_registry.register('blue_commander', 0)
class DeployDecoy(BaseAction):
    """Deploys a generic high-interaction honeypot/decoy service to a target host."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas={f'hosts/{self.target_ip}/decoy': 'active'},
            observation_data={'decoy_deployed': self.target_ip},
        )


@action_registry.register('blue_commander', 1)
class DecoyApache(BaseAction):
    """Deploys a specifically profiled Apache Web Server (Port 80) honeypot."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas={f'hosts/{self.target_ip}/decoy': 'Apache'},
            observation_data={'decoy_deployed': f'Apache on {self.target_ip}'},
        )


@action_registry.register('blue_commander', 2)
class DecoySSHD(BaseAction):
    """Deploys a fake SSH daemon (Port 22) honeypot specifically designed to bait brute force actions."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas={f'hosts/{self.target_ip}/decoy': 'SSHD'},
            observation_data={'decoy_deployed': f'SSHD on {self.target_ip}'},
        )


@action_registry.register('blue_commander', 3)
class DecoyTomcat(BaseAction):
    """Deploys a fake Tomcat server (Port 8080) to deceive application port scans."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas={f'hosts/{self.target_ip}/decoy': 'Tomcat'},
            observation_data={'decoy_deployed': f'Tomcat on {self.target_ip}'},
        )


@action_registry.register('blue_commander', 4)
class Misinform(BaseAction):
    """Injects false host telemetry or alters logging infrastructure to feed Red agents fake data."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas={f'hosts/{self.target_ip}/misinformation': 'active'},
            observation_data={
                'alert': f'Misinformation campaign active on {self.target_ip}.'
            },
        )


@action_registry.register('blue_commander', 5)
class DeployHoneytoken(BaseAction):
    """
    Injects fake, highly-monitored credentials into the memory space of a real host.

    If a Red agent successfully compromises this host and attempts to perform
    post-exploitation (e.g., Pass-the-Hash, credential dumping), they ingest the
    Honeytoken instead. This triggers an immediate, 100% confidence SIEM Alert
    exposing the Red agent's exact location natively.
    """

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(
            agent_id, target_ip=target_ip, cost=5, financial_cost=50, duration=1
        )

    def validate(self, global_state) -> bool:
        return self.target_ip in global_state.all_hosts

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas={f'hosts/{self.target_ip}/contains_honeytokens': True},
            observation_data={
                'alert': f'Honeytokens actively deployed in RAM on {self.target_ip}.'
            },
            eta=self.duration,
        )
