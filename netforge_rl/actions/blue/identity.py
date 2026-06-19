import random
import string
from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.registry import action_registry

from netforge_rl.core.commands import IStateDeltaCommand

class RotateKerberosCommand(IStateDeltaCommand):

    def __init__(self, agent_id):
        self.agent_id = agent_id

    @property
    def target_ip(self):
        return None

    def execute(self, state):
        if self.agent_id in state.agent_funds:
            state.agent_funds[self.agent_id] -= 5000
        state.business_downtime_score += 1500.0
        for agent in state.agent_inventory:
            state.agent_inventory[agent].clear()
        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        new_token = f'Enterprise_Admin_Token_{random_suffix}'

        def _migrate_tokens(token_list):
            if 'Enterprise_Admin_Token' in token_list:
                token_list.remove('Enterprise_Admin_Token')
                token_list.append(new_token)
            old_tokens = [t for t in token_list if t.startswith('Enterprise_Admin_Token_')]
            for t in old_tokens:
                token_list.remove(t)
                token_list.append(new_token)
        for host in state.all_hosts.values():
            _migrate_tokens(host.system_tokens)
            _migrate_tokens(host.cached_credentials)

@action_registry.register('blue_commander', 0)
class RotateKerberos(BaseAction):
    """Rotates Kerberos TGT keys."""

    def __init__(self, agent_id: str, target_ip: str):
        super().__init__(agent_id, target_ip)
        self.duration = 4
        self.compute_cost = 80

    def validate(self, global_state) -> bool:
        if self.agent_id in global_state.agent_funds:
            if global_state.agent_funds[self.agent_id] < 5000:
                return False
        return True

    def execute(self, global_state) -> ActionEffect:
        deltas = {'identity_flush': RotateKerberosCommand(self.agent_id)}
        return ActionEffect(success=True, state_deltas=deltas, observation_data={'alert': 'CRITICAL: Global Domain Keys Rotated. Enterprise Network re-verified.'}, eta=self.duration)