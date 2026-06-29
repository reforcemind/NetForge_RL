import random
from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.registry import action_registry


@action_registry.register('red', 20)
class OverloadPLC(BaseAction):
    """Manipulates PLC setpoints to drive physical parameters into unsafe ranges."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(
            agent_id,
            target_ip=target_ip,
            cost=20,
            financial_cost=1000,
            duration=10,
            required_prior_state='DiscoverNetworkServices',
        )

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
        temp_sp = getattr(
            host, 'temperature_setpoint', getattr(host, 'temperature', 50.0)
        )
        pressure_sp = getattr(
            host, 'pressure_setpoint', getattr(host, 'pressure', 100.0)
        )
        new_temp_sp = temp_sp + random.uniform(80.0, 150.0)
        new_pressure_sp = pressure_sp * random.uniform(1.5, 2.0)
        deltas = {
            f'hosts/{self.target_ip}/temperature_setpoint': new_temp_sp,
            f'hosts/{self.target_ip}/pressure_setpoint': new_pressure_sp,
            f'hosts/{self.target_ip}/flow_rate_setpoint': 3.0,
        }
        obs_data = {
            'action': 'overload_plc',
            'status': 'setpoint_manipulated',
            'temperature_setpoint': new_temp_sp,
            'pressure_setpoint': new_pressure_sp,
        }
        return ActionEffect(
            success=True,
            state_deltas=deltas,
            observation_data=obs_data,
            eta=self.duration,
        )
