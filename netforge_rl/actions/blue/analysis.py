from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.registry import action_registry
from netforge_rl.core.commands import PushSIEMEntryCommand


@action_registry.register('blue', 2)
class Monitor(BaseAction):
    """Forensic subnet sweep: emits SIEM alerts for elevated-privilege or compromised hosts."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip, cost=2, duration=2)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        target_subnet_cidr = None
        if '/' in self.target_ip:
            target_subnet_cidr = self.target_ip
        else:
            host = global_state.all_hosts.get(self.target_ip)
            if host:
                target_subnet_cidr = host.subnet_cidr

        deltas = []
        found = []
        if target_subnet_cidr and target_subnet_cidr in global_state.subnets:
            for ip, host in global_state.subnets[target_subnet_cidr].hosts.items():
                if host.privilege in ('User', 'Root'):
                    deltas.append(
                        PushSIEMEntryCommand(
                            f'[MONITOR] EDR_PRIVILEGE_ALERT target={ip} '
                            f'privilege={host.privilege} compromised_by={host.compromised_by}',
                            target_subnet_cidr,
                        )
                    )
                    found.append(ip)
                elif host.compromised_by != 'None':
                    deltas.append(
                        PushSIEMEntryCommand(
                            f'[MONITOR] EDR_COMPROMISE_ALERT target={ip} '
                            f'compromised_by={host.compromised_by}',
                            target_subnet_cidr,
                        )
                    )
                    found.append(ip)

        return ActionEffect(
            success=True,
            state_deltas=deltas,
            observation_data={
                'monitoring': self.target_ip,
                'alerts_generated': len(found),
            },
        )


@action_registry.register('blue', 3)
class Analyze(BaseAction):
    """Checks host for IoCs."""

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
        knowledge_deltas = {f'knowledge/{self.agent_id}/{self.target_ip}': 'True'}
        return ActionEffect(
            success=True, state_deltas=knowledge_deltas, observation_data=obs_data
        )
