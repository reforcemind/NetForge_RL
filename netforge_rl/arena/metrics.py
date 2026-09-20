from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from netforge_rl.environment.constants import PADDING_SUBNET


@dataclass
class ArenaMetrics:
    """Per-episode competition metrics. Reward is recorded, not the ranking axis."""

    mission_success: float
    disruption: float
    security: float
    false_positives: float
    exfiltration: float
    compromised_hosts: float
    isolated_hosts: float
    sla_uptime: float
    catastrophic_failure: float
    blue_return: float
    red_return: float
    steps: int
    detection_rate: float = 0.0
    mttc: float = 0.0
    deception_efficacy: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def metrics_from_env(
    env, last_info: dict, blue_return: float, red_return: float
) -> ArenaMetrics:
    sample = next((v for k, v in last_info.items() if 'blue' in k), {})
    if not sample:
        sample = next(iter(last_info.values()), {})
    active = [
        h
        for h in env.global_state.all_hosts.values()
        if h.subnet_cidr != PADDING_SUBNET
    ]
    total = max(len(active), 1)
    kinetic = any(
        getattr(h, 'system_integrity', 'clean') == 'kinetic_destruction' for h in active
    )
    sla = float(sample.get('SLA_Uptime_Percentage', 1.0))
    compromised = float(sample.get('compromised_hosts', 0.0))
    isolated = float(sample.get('isolated_hosts', 0.0))
    fps = float(sample.get('false_positives_total', sample.get('false_positives', 0.0)))
    return ArenaMetrics(
        mission_success=float((not kinetic) and sla >= 0.5),
        disruption=isolated / total,
        security=max(0.0, 1.0 - compromised / total),
        false_positives=fps,
        exfiltration=float(sample.get('Total_Exfiltrated_Data', 0.0)),
        compromised_hosts=compromised,
        isolated_hosts=isolated,
        sla_uptime=sla,
        catastrophic_failure=float(kinetic),
        blue_return=float(blue_return),
        red_return=float(red_return),
        steps=int(getattr(env, 'current_tick', 0)),
        detection_rate=float(sample.get('detection_rate', 0.0)),
        mttc=float(sample.get('MTTC', 0.0)),
        deception_efficacy=float(sample.get('deception_efficacy', 0.0)),
    )


def cvar(values: list[float], alpha: float = 0.05) -> float:
    """Lower-tail Conditional Value at Risk (expected value in the worst α)."""
    if not values:
        return 0.0
    arr = np.sort(np.asarray(values, dtype=float))
    k = max(1, int(np.ceil(alpha * len(arr))))
    return float(arr[:k].mean())


def worst_case(values: list[float]) -> float:
    return float(min(values)) if values else 0.0


def tail_failure_rate(flags: list[float], threshold: float = 0.5) -> float:
    if not flags:
        return 0.0
    arr = np.asarray(flags, dtype=float)
    return float(np.mean(arr >= threshold))


@dataclass
class AggregateReport:
    n_episodes: int
    mean: dict = field(default_factory=dict)
    std: dict = field(default_factory=dict)
    cvar05: dict = field(default_factory=dict)
    worst: dict = field(default_factory=dict)
    catastrophic_rate: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


_SCALAR_KEYS = (
    'mission_success',
    'disruption',
    'security',
    'false_positives',
    'exfiltration',
    'compromised_hosts',
    'isolated_hosts',
    'sla_uptime',
    'catastrophic_failure',
    'blue_return',
    'red_return',
    'detection_rate',
    'mttc',
)


def aggregate(metrics: list[ArenaMetrics], alpha: float = 0.05) -> AggregateReport:
    if not metrics:
        return AggregateReport(n_episodes=0)
    series = {k: [float(getattr(m, k)) for m in metrics] for k in _SCALAR_KEYS}
    return AggregateReport(
        n_episodes=len(metrics),
        mean={k: round(float(np.mean(v)), 4) for k, v in series.items()},
        std={
            k: round(float(np.std(v, ddof=1) if len(v) > 1 else 0.0), 4)
            for k, v in series.items()
        },
        cvar05={k: round(cvar(v, alpha), 4) for k, v in series.items()},
        worst={k: round(worst_case(v), 4) for k, v in series.items()},
        catastrophic_rate=round(
            tail_failure_rate(series['catastrophic_failure'], 0.5), 4
        ),
    )
