from __future__ import annotations

from dataclasses import dataclass, replace

from netforge_rl.environment.config.schema import (
    DifficultyName,
    EnvConfig,
    TopologyRates,
)

EVAL_SEEDS: tuple[int, ...] = tuple(range(9001, 9021))


@dataclass(frozen=True)
class DifficultyPreset:
    name: DifficultyName
    max_active_hosts: int
    dhcp_interval: int
    topology: TopologyRates
    log_latency: int
    max_ticks: int = 200

    def apply(self, cfg: EnvConfig) -> EnvConfig:
        return replace(
            cfg,
            max_active_hosts=self.max_active_hosts,
            dhcp_interval=self.dhcp_interval,
            topology=self.topology,
            log_latency=self.log_latency,
            max_ticks=self.max_ticks,
        )


DIFFICULTIES: dict[str, DifficultyPreset] = {
    'easy': DifficultyPreset(
        name='easy',
        max_active_hosts=6,
        dhcp_interval=0,
        topology=TopologyRates(),
        log_latency=0,
    ),
    'medium': DifficultyPreset(
        name='medium',
        max_active_hosts=15,
        dhcp_interval=80,
        topology=TopologyRates(churn=0.01),
        log_latency=2,
    ),
    'hard': DifficultyPreset(
        name='hard',
        max_active_hosts=100,
        dhcp_interval=40,
        topology=TopologyRates(churn=0.02, migration=0.01, arrival=0.005),
        log_latency=4,
    ),
}

DIFFICULTY_PRESETS: dict[str, dict] = {
    name: preset.apply(EnvConfig()).to_dict() for name, preset in DIFFICULTIES.items()
}
