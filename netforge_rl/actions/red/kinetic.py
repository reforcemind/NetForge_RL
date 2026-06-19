import random
from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.registry import action_registry

@action_registry.register('red_operator', 20)
class OverloadPLC(BaseAction):
    """Spikes OT node temperature."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip=target_ip, cost=20, financial_cost=1000, duration=10, required_prior_state='DiscoverNetworkServices')

    def validate(self, global_state) -> bool:
        if not super().validate(global_state):
            return False
        host = global_state.all_hosts.get(self.target_ip)
        if not host:
            return False
        if host.os != 'PLC_Firmware' or host.subnet_cidr != '10.0.99.0/24':
            return False
        if host.privilege != 'Root':
            return False
        return global_state.can_route_to(self.target_ip, agent_id=self.agent_id)

    def execute(self, global_state) -> ActionEffect:
        host = global_state.all_hosts.get(self.target_ip)
        if not host:
            return ActionEffect(success=False, state_deltas={}, observation_data={})
        current_temperature = getattr(host, 'temperature', 50.0)
        spike = random.uniform(90.0, 150.0)
        new_temp = current_temperature + spike
        deltas = {f'hosts/{self.target_ip}/temperature': new_temp, f'hosts/{self.target_ip}/system_integrity': 'kinetic_destruction'}
        obs_data = {'action': 'overload_plc', 'status': 'kinetic_impact_achieved', 'terminal_temperature': new_temp}
        return ActionEffect(success=True, state_deltas=deltas, observation_data=obs_data, eta=self.duration)