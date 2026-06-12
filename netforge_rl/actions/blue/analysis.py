from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.registry import action_registry


@action_registry.register('blue_operator', 2)
class Monitor(BaseAction):
    """Deploys active traffic analysis scanning on a specific subnet or host."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        knowledge_deltas = {}
        target_subnet_cidr = None

        # Determine if target is a specific IP or CIDR block
        if '/' in self.target_ip:
            target_subnet_cidr = self.target_ip
        else:
            host = global_state.all_hosts.get(self.target_ip)
            if host:
                target_subnet_cidr = host.subnet_cidr

        if target_subnet_cidr and target_subnet_cidr in global_state.subnets:
            subnet_hosts = global_state.subnets[target_subnet_cidr].hosts
            for ip, host in subnet_hosts.items():
                if host.edr_active:
                    knowledge_deltas[f'knowledge/{self.agent_id}/{ip}'] = 'True'

        return ActionEffect(
            success=True,
            state_deltas=knowledge_deltas,
            observation_data={'monitoring': self.target_ip},
        )


@action_registry.register('blue_operator', 3)
class Analyze(BaseAction):
    """Executes a forensic deep scan of a specific host for malware indicators."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        obs_data = {'analysis_report': self.target_ip}

        if self.target_ip in global_state.all_hosts:
            host = global_state.all_hosts[self.target_ip]

            if not host.edr_active:
                return ActionEffect(
                    success=False,
                    state_deltas={},
                    observation_data={'error': 'EDR blindspot - telemetry unavailable'},
                )

            if host.privilege in ['User', 'Root']:
                obs_data['ioc_found'] = True
                obs_data['compromised_by'] = host.compromised_by
                obs_data['exact_privilege'] = host.privilege
            else:
                obs_data['ioc_found'] = False

        # Analysis guarantees Blue Team knowledge of this precise host
        knowledge_deltas = {f'knowledge/{self.agent_id}/{self.target_ip}': 'True'}

        return ActionEffect(
            success=True,
            state_deltas=knowledge_deltas,
            observation_data=obs_data,
        )
