from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Tuple
from netforge_rl.core.state import Firewall


def iter_host_deltas(state_deltas: Any) -> Iterator[Tuple[str, str, Any]]:
    """Yield ``(attribute, ip, value)`` host mutations from either delta encoding.
    """
    if isinstance(state_deltas, dict):
        for key, value in state_deltas.items():
            parts = key.split('/')
            if len(parts) == 3 and parts[0] == 'hosts':
                yield parts[2], parts[1], value
    elif isinstance(state_deltas, list):
        for cmd in state_deltas:
            name = type(cmd).__name__
            if name == 'UpdateHostPrivilegeCommand':
                yield 'privilege', cmd.target_ip, cmd.privilege
                if cmd.compromised_by:
                    yield 'compromised_by', cmd.target_ip, cmd.compromised_by
            elif name == 'UpdateHostStatusCommand':
                yield 'status', cmd.target_ip, cmd.status


class IStateDeltaCommand(ABC):
    """Object-oriented state mutation; the resolver reads target_ip."""

    @abstractmethod
    def execute(self, global_state: Any): ...

    @property
    @abstractmethod
    def target_ip(self) -> Optional[str]: ...


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


class BlockPortCommand(IStateDeltaCommand):
    def __init__(self, subnet: str, port: int):
        self.subnet = subnet
        self.port = port

    @property
    def target_ip(self):
        return None

    def execute(self, global_state):
        global_state.firewalls.setdefault('global', Firewall('global')).block_port(
            self.subnet, self.port
        )


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


class PushSIEMEntryCommand(IStateDeltaCommand):
    """Pushes a structured log entry into the SIEM buffer from an action effect."""

    def __init__(self, log_line: str, subnet_cidr: str):
        self.log_line = log_line
        self.subnet_cidr = subnet_cidr

    @property
    def target_ip(self):
        return None

    def execute(self, global_state):
        global_state.siem_log_buffer.append((self.log_line, self.subnet_cidr))
        if len(global_state.siem_log_buffer) > 64:
            global_state.siem_log_buffer.pop(0)


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
