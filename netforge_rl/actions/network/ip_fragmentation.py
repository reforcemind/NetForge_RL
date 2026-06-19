from typing import TYPE_CHECKING
from netforge_rl.core.action import BaseAction, ActionEffect
if TYPE_CHECKING:
    from netforge_rl.core.state import GlobalNetworkState
from netforge_rl.core.registry import action_registry

@action_registry.register('red_operator', 11)
class IPFragmentationAction(BaseAction):
    """Fragments payload to bypass IDS."""

    def __init__(self, agent_id: str, target_ip: str, payload_type: str='reverse_shell'):
        super().__init__(agent_id=agent_id, target_ip=target_ip)
        self.payload_type = payload_type

    def validate(self, global_state: 'GlobalNetworkState') -> bool:
        return global_state.can_route_to(self.target_ip, agent_id=self.agent_id)

    def execute(self, global_state: 'GlobalNetworkState') -> ActionEffect:
        target_host = global_state.all_hosts.get(self.target_ip)
        if not target_host:
            return ActionEffect(success=False, state_deltas={}, observation_data={'error': 'Host not found'})

        ids_present = 'IDS' in target_host.services
        success = False
        state_deltas = {}
        observation_data = {}
        if ids_present:
            observation_data['alert'] = 'IDS_SIGNATURE_IP_FRAGMENTATION_DETECTED'
            success = False
        else:
            state_deltas[f'hosts/{self.target_ip}/privilege'] = 'User'
            state_deltas[f'hosts/{self.target_ip}/compromised_by'] = self.agent_id
            success = True
        return ActionEffect(success=success, state_deltas=state_deltas, observation_data=observation_data)