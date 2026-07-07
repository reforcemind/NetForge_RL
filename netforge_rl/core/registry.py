import inspect
from typing import Callable, Dict, Optional, Type


def team_of(agent_id: str) -> str:
    """Every agent belongs to exactly one action team: 'red' or 'blue'."""
    return 'red' if 'red' in agent_id.lower() else 'blue'


class ActionRegistry:
    """Factory registry mapping ``(team, group_id) -> BaseAction subclass."""

    def __init__(self):
        self._actions: Dict[str, Dict[int, Type]] = {'red': {}, 'blue': {}}

    def register(self, team: str, group_id: int) -> Callable:
        def decorator(cls):
            self._actions.setdefault(team, {})[group_id] = cls
            return cls

        return decorator

    def get_action_class(self, agent_id: str, group_id: int) -> Optional[Type]:
        return self._actions.get(team_of(agent_id), {}).get(group_id)

    def instantiate_action(
        self, agent_id: str, action_data: object, target_ips: list
    ) -> Optional[object]:
        """Resolve an action payload to a BaseAction instance."""
        if not target_ips:
            target_ips = ['127.0.0.1']
        if (
            isinstance(action_data, (list, tuple))
            or type(action_data).__name__ == 'ndarray'
        ):
            action_type_id = int(action_data[0])
            target_ip = target_ips[int(action_data[1]) % len(target_ips)]
        else:
            action_int = int(action_data)
            target_ip = target_ips[action_int % len(target_ips)]
            action_type_id = action_int // len(target_ips) % 12
        ActionCls = self.get_action_class(agent_id, action_type_id)
        if not ActionCls:
            return None
        params = inspect.signature(ActionCls.__init__).parameters
        kwargs = {'agent_id': agent_id}
        if 'target_ip' in params:
            kwargs['target_ip'] = target_ip
        elif 'target_subnet' in params:
            a, b, c = target_ip.split('.')[:3]
            kwargs['target_subnet'] = f'{a}.{b}.{c}.0/24'
        elif 'target_agent_id' in params:
            kwargs['target_agent_id'] = (
                'red_operator' if agent_id == 'red_commander' else 'red_commander'
            )
        return ActionCls(**kwargs)


action_registry = ActionRegistry()
