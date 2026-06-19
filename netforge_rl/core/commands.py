from abc import ABC, abstractmethod
from typing import Any, Optional


class IStateDeltaCommand(ABC):
    """Object-oriented state mutation; the resolver reads target_ip."""

    @abstractmethod
    def execute(self, global_state: Any): ...

    @property
    @abstractmethod
    def target_ip(self) -> Optional[str]: ...


class UpdateKnowledgeCommand(IStateDeltaCommand):
    def __init__(self, agent_id: str, ip: str, value: Any = True):
        self.agent_id = agent_id
        self._target_ip = ip
        self.value = value

    @property
    def target_ip(self):
        return self._target_ip

    def execute(self, global_state):
        global_state.update_knowledge(self.agent_id, self.target_ip)


class UpdateHostPrivilegeCommand(IStateDeltaCommand):
    def __init__(self, ip: str, privilege: str, compromised_by: Optional[str] = None):
        self._target_ip = ip
        self.privilege = privilege
        self.compromised_by = compromised_by

    @property
    def target_ip(self):
        return self._target_ip

    def execute(self, global_state):
        host = global_state.all_hosts.get(self._target_ip)
        if host is None:
            return
        host.privilege = self.privilege
        if self.compromised_by:
            host.compromised_by = self.compromised_by


class UpdateHostStatusCommand(IStateDeltaCommand):
    def __init__(self, ip: str, status: str):
        self._target_ip = ip
        self.status = status

    @property
    def target_ip(self):
        return self._target_ip

    def execute(self, global_state):
        host = global_state.all_hosts.get(self._target_ip)
        if host is not None:
            host.status = self.status


class UpdateServiceCommand(IStateDeltaCommand):
    def __init__(self, ip: str, service: str, action: str = 'remove'):
        self._target_ip = ip
        self.service = service
        self.action = action

    @property
    def target_ip(self):
        return self._target_ip

    def execute(self, global_state):
        host = global_state.all_hosts.get(self._target_ip)
        if host is None:
            return
        if self.action == 'remove' and self.service in host.services:
            host.services.remove(self.service)
        elif self.action == 'add' and self.service not in host.services:
            host.services.append(self.service)


class BlockPortCommand(IStateDeltaCommand):
    def __init__(self, subnet: str, port: int):
        self.subnet = subnet
        self.port = port

    @property
    def target_ip(self):
        return None

    def execute(self, global_state):
        from netforge_rl.core.state import Firewall

        global_state.firewalls.setdefault('global', Firewall('global')).block_port(
            self.subnet, self.port
        )


class AddHistoryCommand(IStateDeltaCommand):
    def __init__(self, agent_id: str, record: str):
        self.agent_id = agent_id
        self.record = record

    @property
    def target_ip(self):
        return None

    def execute(self, global_state):
        global_state.action_history.setdefault(self.agent_id, set()).add(self.record)


class UpdateDecoyCommand(IStateDeltaCommand):
    def __init__(self, ip: str, decoy_type: str):
        self._target_ip = ip
        self.decoy_type = decoy_type

    @property
    def target_ip(self):
        return self._target_ip

    def execute(self, global_state):
        host = global_state.all_hosts.get(self._target_ip)
        if host is not None:
            host.decoy = self.decoy_type


class EstablishSessionCommand(IStateDeltaCommand):
    def __init__(self, agent_id: str, ip: str, port: int):
        self.agent_id = agent_id
        self._target_ip = ip
        self.port = port

    @property
    def target_ip(self):
        return self._target_ip

    def execute(self, global_state):
        global_state.active_sessions.setdefault(self.agent_id, []).append(
            {'ip': self._target_ip, 'port': self.port}
        )


class DropSessionCommand(IStateDeltaCommand):
    def __init__(self, ip: str):
        self._target_ip = ip

    @property
    def target_ip(self):
        return self._target_ip

    def execute(self, global_state):
        for agent_id, sessions in global_state.active_sessions.items():
            global_state.active_sessions[agent_id] = [
                s for s in sessions if s['ip'] != self._target_ip
            ]


class ConsumeBandwidthCommand(IStateDeltaCommand):
    """Volumetric SIEM trigger: subnet usage > 1000/tick raises a high-severity alert."""

    def __init__(self, subnet: str, amount: int):
        self.subnet = subnet
        self.amount = amount

    @property
    def target_ip(self):
        return None

    def execute(self, global_state):
        global_state.subnet_bandwidth[self.subnet] = (
            global_state.subnet_bandwidth.get(self.subnet, 0) + self.amount
        )
        if global_state.subnet_bandwidth[self.subnet] > 1000:
            alert = {
                'type': 'volumetric_anomaly',
                'subnet': self.subnet,
                'severity': 'High',
            }
            if alert not in global_state.siem_log_buffer:
                global_state.siem_log_buffer.append(alert)
