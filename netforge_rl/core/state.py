import random
from typing import Any, Dict, Set
import numpy as np


class Host:
    def __init__(self, ip: str, hostname: str, subnet_cidr: str):
        self.ip = ip
        self.hostname = hostname
        self.subnet_cidr = subnet_cidr
        self.status = 'online'
        self.privilege = 'None'
        self.decoy = 'inactive'
        self.compromised_by = 'None'
        self.edr_active = False
        self.os = 'Unknown'
        self.services: list = []
        self.vulnerabilities: list = []
        self.is_domain_controller = False
        self.human_vulnerability_score = 0.5
        self.contains_honeytokens = False
        self.misinformation = False
        self.cached_credentials: list = []
        self.system_tokens: list = []

    def __repr__(self):
        return (
            f'<Host {self.ip} | Priv: {self.privilege} | Breach: {self.compromised_by}>'
        )


class Subnet:
    def __init__(self, cidr: str, name: str):
        self.cidr = cidr
        self.name = name
        self.hosts: Dict[str, Host] = {}

    def add_host(self, host: Host):
        self.hosts[host.ip] = host


class Firewall:
    def __init__(self, name: str):
        self.name = name
        self.rules: Dict[tuple[str, int], str] = {}

    def block_port(self, target_subnet: str, port: int):
        self.rules[target_subnet, port] = 'block'

    def is_blocked(self, target_subnet: str, port: int) -> bool:
        return self.rules.get((target_subnet, port)) == 'block'


class GlobalNetworkState:
    """Mutable single source of truth for the legacy MARL physics engine."""

    def __init__(self):
        self.subnets: Dict[str, Subnet] = {}
        self.all_hosts: Dict[str, Host] = {}
        self.firewalls: Dict[str, Firewall] = {}
        self.agent_knowledge: Dict[str, Set[str]] = {}
        self.agent_inventory: Dict[str, set] = {}
        self.agent_energy: Dict[str, int] = {}
        self.agent_funds: Dict[str, int] = {}
        self.agent_compute: Dict[str, int] = {}
        self.business_downtime_score = 0.0
        self.agent_locked_until: Dict[str, int] = {}
        self.action_history: Dict[str, set] = {}
        self.siem_log_buffer: list = []
        self.current_tick = 0
        self.active_sessions: Dict[str, list] = {}
        self.subnet_bandwidth: Dict[str, int] = {}

    def update_knowledge(self, agent_id: str, ip: str):
        self.agent_knowledge.setdefault(agent_id, set()).add(ip)

    def add_subnet(self, subnet: Subnet):
        self.subnets[subnet.cidr] = subnet

    def register_host(self, host: Host):
        self.all_hosts[host.ip] = host
        if host.subnet_cidr in self.subnets:
            self.subnets[host.subnet_cidr].add_host(host)

    def apply_delta(self, delta_key: Any, delta_value: Any = None):
        """Apply a state delta — either a Command object or a string."""
        if hasattr(delta_key, 'execute') and callable(delta_key.execute):
            delta_key.execute(self)
            return
        if not isinstance(delta_key, str):
            return
        parts = delta_key.split('/')
        if parts[0] == 'hosts' and len(parts) == 3:
            ip, attribute = (parts[1], parts[2])
            host = self.all_hosts.get(ip)
            if host is not None and hasattr(host, attribute):
                setattr(host, attribute, delta_value)
        elif parts[0] == 'knowledge' and len(parts) == 3:
            self.update_knowledge(parts[1], parts[2])
        elif parts[0] == 'firewall' and parts[1] == 'block' and (len(parts) == 4):
            subnet = parts[2].replace('_slash_', '/')
            self.firewalls.setdefault('global', Firewall('global')).block_port(
                subnet, int(parts[3])
            )
        elif parts[0] == 'history' and len(parts) == 3:
            self.action_history.setdefault(parts[1], set()).add(parts[2])

    def get_subnet_name(self, cidr: str) -> str:
        subnet = self.subnets.get(cidr)
        return subnet.name if subnet else 'Unknown'

    def can_route_to(
        self, target_ip: str, port: int = None, agent_id: str = None
    ) -> bool:
        """Evaluate subnet routing + firewall blocks + ZTNA gate."""
        host = self.all_hosts.get(target_ip)
        if host is None or host.status == 'isolated':
            return False
        target_subnet = host.subnet_cidr
        if port is not None and any(
            (fw.is_blocked(target_subnet, port) for fw in self.firewalls.values())
        ):
            return False
        subnet_name = self.get_subnet_name(target_subnet)
        if subnet_name == 'DMZ':
            return True
        has_dmz_pivot = any(
            (
                h.privilege in ('User', 'Root')
                and self.get_subnet_name(h.subnet_cidr) == 'DMZ'
                for h in self.all_hosts.values()
            )
        )
        if subnet_name == 'Corporate':
            return has_dmz_pivot
        if subnet_name == 'Secure':
            has_corp_pivot = any(
                (
                    h.privilege in ('User', 'Root')
                    and self.get_subnet_name(h.subnet_cidr) == 'Corporate'
                    for h in self.all_hosts.values()
                )
            )
            if not (has_dmz_pivot or has_corp_pivot):
                return False
            if agent_id and agent_id.startswith('red'):
                inv = self.agent_inventory.get(agent_id, set())
                if 'Enterprise_Admin_Token' not in inv:
                    return False
            return True
        return False

    def get_adjacency_matrix(self) -> np.ndarray:
        """100x100 adjacency matrix; ``can_route_to`` is destination-only so rows broadcast the same decision."""
        adj = np.zeros((100, 100), dtype=np.float32)
        sorted_ips = sorted(self.all_hosts.keys())
        for i, _src_ip in enumerate(sorted_ips):
            for j, dst_ip in enumerate(sorted_ips):
                if i == j or self.can_route_to(dst_ip):
                    adj[i, j] = 1.0
        return adj

    def reallocate_dhcp(self):
        """Reshuffle IPs on every non-DMZ subnet; invalidates stale agent knowledge."""
        for subnet in self.subnets.values():
            if subnet.name == 'DMZ':
                continue
            hosts = list(subnet.hosts.values())
            if not hosts:
                continue
            base_ip = subnet.cidr.split('.0/')[0]
            new_ips = random.sample(range(1, 250), len(hosts))
            new_subnet_hosts = {}
            for i, host in enumerate(hosts):
                self.all_hosts.pop(host.ip, None)
                host.ip = f'{base_ip}.{new_ips[i]}'
                self.all_hosts[host.ip] = host
                new_subnet_hosts[host.ip] = host
            subnet.hosts = new_subnet_hosts
