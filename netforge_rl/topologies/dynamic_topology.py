from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from netforge_rl.core.state import GlobalNetworkState


@dataclass
class TopologyEvent:
    kind: str  # 'host_offline' | 'host_online' | 'host_migrate' | 'host_arrive'
    ip: str  # current IP after event
    detail: dict


class TopologyEventEngine:
    """Emits mid-episode topology changes: churn, migration, and device arrival.
    """

    def __init__(
        self,
        churn_rate: float = 0.02,
        migration_rate: float = 0.01,
        arrival_rate: float = 0.005,
    ):
        self.churn_rate = churn_rate
        self.migration_rate = migration_rate
        self.arrival_rate = arrival_rate
        self._rng = random.Random()

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def tick(self, state: GlobalNetworkState) -> List[TopologyEvent]:
        events: List[TopologyEvent] = []

        active = [
            h
            for h in state.all_hosts.values()
            if h.status == 'online' and not h.ip.startswith('169.254')
        ]
        offline_real = [
            h
            for h in state.all_hosts.values()
            if h.status == 'isolated' and not h.ip.startswith('169.254')
        ]
        padding = [h for h in state.all_hosts.values() if h.ip.startswith('169.254')]

        for host in active:
            if self._rng.random() < self.churn_rate:
                host.status = 'isolated'
                events.append(
                    TopologyEvent('host_offline', host.ip, {'subnet': host.subnet_cidr})
                )

        for host in offline_real:
            if self._rng.random() < self.churn_rate * 2.0:
                host.status = 'online'
                events.append(
                    TopologyEvent('host_online', host.ip, {'subnet': host.subnet_cidr})
                )

        if active and self._rng.random() < self.migration_rate:
            host = self._rng.choice(active)
            moveable = [
                s
                for s in state.subnets.values()
                if s.name not in ('DMZ', 'OT_Subnet') and s.cidr != host.subnet_cidr
            ]
            if moveable:
                target = self._rng.choice(moveable)
                new_ip = self._free_ip(target.cidr, state)
                if new_ip:
                    old_ip, old_cidr = host.ip, host.subnet_cidr
                    state.all_hosts.pop(old_ip)
                    state.subnets[old_cidr].hosts.pop(old_ip, None)
                    host.ip = new_ip
                    host.subnet_cidr = target.cidr
                    state.all_hosts[new_ip] = host
                    target.hosts[new_ip] = host
                    for known in state.agent_knowledge.values():
                        known.discard(old_ip)
                    events.append(
                        TopologyEvent(
                            'host_migrate',
                            new_ip,
                            {
                                'old_ip': old_ip,
                                'old_subnet': old_cidr,
                                'new_subnet': target.cidr,
                            },
                        )
                    )

        if padding and self._rng.random() < self.arrival_rate:
            pad = self._rng.choice(padding)
            joinable = [
                s
                for s in state.subnets.values()
                if s.name not in ('DMZ', 'OT_Subnet') and '169.254' not in s.cidr
            ]
            if joinable:
                target = self._rng.choice(joinable)
                new_ip = self._free_ip(target.cidr, state)
                if new_ip:
                    state.all_hosts.pop(pad.ip)
                    pad.ip = new_ip
                    pad.subnet_cidr = target.cidr
                    pad.status = 'online'
                    pad.hostname = f'BYOD_{new_ip.split(".")[-1]}'
                    pad.os = self._rng.choice(['Windows_10', 'Linux_Ubuntu'])
                    pad.services = ['RDP'] if 'Windows' in pad.os else ['SSH']
                    state.all_hosts[new_ip] = pad
                    target.hosts[new_ip] = pad
                    events.append(
                        TopologyEvent('host_arrive', new_ip, {'subnet': target.cidr})
                    )

        return events

    def _free_ip(self, cidr: str, state: GlobalNetworkState) -> str | None:
        base = cidr.split('.0/')[0]
        occupied = set(state.all_hosts)
        for _ in range(30):
            candidate = f'{base}.{self._rng.randint(10, 250)}'
            if candidate not in occupied:
                return candidate
        return None
