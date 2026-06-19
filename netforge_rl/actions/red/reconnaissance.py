from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.registry import action_registry


@action_registry.register('red_commander', 0)
class NetworkScan(BaseAction):
    """Maps active IPs on subnet."""

    def __init__(self, agent_id: str, target_subnet: str):
        super().__init__(agent_id, target_ip=target_subnet, cost=5)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        return ActionEffect(
            success=True,
            state_deltas={},
            observation_data={'discovered_subnet': self.target_ip},
            eta=3,
        )


@action_registry.register('red_commander', 1)
class DiscoverRemoteSystems(BaseAction):
    """Executes ping sweep."""

    def __init__(self, agent_id: str, target_subnet: str):
        super().__init__(agent_id, target_ip=target_subnet, cost=3)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        fake_data = False
        active_hosts = []
        for host in global_state.all_hosts.values():
            if host.subnet_cidr == self.target_ip:
                active_hosts.append(host.ip)
                if host.decoy in ['active', 'Apache', 'SSHD', 'Tomcat']:
                    fake_data = True
        obs_data = {'ping_sweep': self.target_ip, 'hosts': active_hosts}
        if fake_data:
            obs_data['hosts'] = ['10.x.x.99', '10.x.x.100']
        knowledge_deltas = {
            f'knowledge/{self.agent_id}/{ip}': 'True' for ip in obs_data['hosts']
        }
        return ActionEffect(
            success=True, state_deltas=knowledge_deltas, observation_data=obs_data
        )


@action_registry.register('red_commander', 2)
@action_registry.register('red', 2)
class DiscoverNetworkServices(BaseAction):
    """Executes port scan."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip, cost=2, duration=3)

    def validate(self, global_state) -> bool:
        return True

    def execute(self, global_state) -> ActionEffect:
        obs_data = {'port_scan': self.target_ip}
        if self.target_ip in global_state.all_hosts:
            host = global_state.all_hosts[self.target_ip]
            if host.decoy in ['active', 'Apache']:
                obs_data['services'] = ['Fake_Apache_80', 'Fake_SSH_2222']
            elif host.decoy == 'SSHD':
                obs_data['services'] = ['Fake_SSH_22']
            elif host.decoy == 'Tomcat':
                obs_data['services'] = ['Fake_Tomcat_8080']
            else:
                obs_data['services'] = host.services
                obs_data['os'] = host.os
                obs_data['vulnerabilities'] = host.vulnerabilities
        knowledge_deltas = {
            f'knowledge/{self.agent_id}/{self.target_ip}': 'True',
            f'history/{self.agent_id}/DiscoverNetworkServices:{self.target_ip}': 'add',
        }
        return ActionEffect(
            success=True, state_deltas=knowledge_deltas, observation_data=obs_data
        )
