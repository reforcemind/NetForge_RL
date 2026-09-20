from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from netforge_rl.environment.config.agents import AgentIds

TimeMode = Literal['event', 'fixed']
DifficultyName = Literal['easy', 'medium', 'hard']

_FLAT_KEYS = frozenset(
    {
        'scenario_type',
        'max_ticks',
        'max_active_hosts',
        'evaluation_mode',
        'topology_path',
        'topology_spec',
        'log_latency',
        'dhcp_interval',
        'record_siem',
        'record_trajectory',
        'pcap_obs',
        'time_mode',
        'docker_mode',
        'nlp_backend',
        'topology_churn_rate',
        'topology_migration_rate',
        'topology_arrival_rate',
    }
)


@dataclass(frozen=True)
class TopologyRates:
    churn: float = 0.0
    migration: float = 0.0
    arrival: float = 0.0


@dataclass(frozen=True)
class ResourceBudgets:
    energy: int = 50
    blue_funds: int = 10_000
    red_funds: int = 5_000
    compute: int = 1_000


@dataclass(frozen=True)
class EnvConfig:
    """Typed env knobs. ``NetForgeRLEnv`` still accepts a dict and parses this."""

    scenario_type: str = 'ransomware'
    max_ticks: int = 1000
    max_active_hosts: int | None = None
    evaluation_mode: bool = False
    topology_path: str | None = None
    topology_spec: dict | None = None
    log_latency: int = 0
    dhcp_interval: int = 40
    record_siem: bool = False
    record_trajectory: bool = False
    pcap_obs: bool = False
    time_mode: TimeMode = 'event'
    docker_mode: str = 'sim'
    nlp_backend: str = 'tfidf'
    topology: TopologyRates = field(default_factory=TopologyRates)
    agents: AgentIds = field(default_factory=AgentIds)
    budgets: ResourceBudgets = field(default_factory=ResourceBudgets)
    extras: dict[str, Any] = field(default_factory=dict)

    def replace(self, **changes) -> EnvConfig:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'scenario_type': self.scenario_type,
            'max_ticks': self.max_ticks,
            'max_active_hosts': self.max_active_hosts,
            'evaluation_mode': self.evaluation_mode,
            'topology_path': self.topology_path,
            'topology_spec': self.topology_spec,
            'log_latency': self.log_latency,
            'dhcp_interval': self.dhcp_interval,
            'record_siem': self.record_siem,
            'record_trajectory': self.record_trajectory,
            'pcap_obs': self.pcap_obs,
            'time_mode': self.time_mode,
            'docker_mode': self.docker_mode,
            'nlp_backend': self.nlp_backend,
            'topology_churn_rate': self.topology.churn,
            'topology_migration_rate': self.topology.migration,
            'topology_arrival_rate': self.topology.arrival,
        }
        payload.update(self.extras)
        return payload

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | EnvConfig | None = None
    ) -> EnvConfig:
        if data is None:
            return cls()
        if isinstance(data, EnvConfig):
            return data
        raw = dict(data)
        topo = TopologyRates(
            churn=float(raw.pop('topology_churn_rate', 0.0) or 0.0),
            migration=float(raw.pop('topology_migration_rate', 0.0) or 0.0),
            arrival=float(raw.pop('topology_arrival_rate', 0.0) or 0.0),
        )
        known: dict[str, Any] = {}
        for key in (
            'scenario_type',
            'max_ticks',
            'max_active_hosts',
            'evaluation_mode',
            'topology_path',
            'topology_spec',
            'log_latency',
            'dhcp_interval',
            'record_siem',
            'record_trajectory',
            'pcap_obs',
            'time_mode',
            'docker_mode',
            'nlp_backend',
        ):
            if key in raw:
                known[key] = raw.pop(key)
        extras = {k: v for k, v in raw.items() if k not in _FLAT_KEYS}
        return cls(topology=topo, extras=extras, **known)
