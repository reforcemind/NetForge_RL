from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from netforge_rl.environment.parallel_env import NetForgeRLEnv


@dataclass
class PhaseConfig:
    name: str
    max_active_hosts: int
    scenario_types: List[str]
    reward_scale: float
    dhcp_interval: int  # ticks between DHCP reshuffles; 0 = disabled
    topology_churn_rate: float
    topology_migration_rate: float
    topology_arrival_rate: float
    advance_threshold: Optional[
        float
    ]  # mean total reward to graduate; None = final phase
    advance_window: int  # episodes in the rolling window


PHASES: List[PhaseConfig] = [
    PhaseConfig(
        name='novice',
        max_active_hosts=5,
        scenario_types=['ransomware'],
        reward_scale=3.0,
        dhcp_interval=0,
        topology_churn_rate=0.0,
        topology_migration_rate=0.0,
        topology_arrival_rate=0.0,
        advance_threshold=60.0,
        advance_window=10,
    ),
    PhaseConfig(
        name='intermediate',
        max_active_hosts=25,
        scenario_types=['ransomware', 'apt_espionage'],
        reward_scale=1.5,
        dhcp_interval=80,
        topology_churn_rate=0.0,
        topology_migration_rate=0.0,
        topology_arrival_rate=0.0,
        advance_threshold=40.0,
        advance_window=15,
    ),
    PhaseConfig(
        name='expert',
        max_active_hosts=100,
        scenario_types=['ransomware', 'apt_espionage', 'iot_grid'],
        reward_scale=1.0,
        dhcp_interval=40,
        topology_churn_rate=0.02,
        topology_migration_rate=0.01,
        topology_arrival_rate=0.005,
        advance_threshold=None,
        advance_window=20,
    ),
]


def _build_env(
    phase: PhaseConfig, base_cfg: dict, seed: Optional[int]
) -> NetForgeRLEnv:
    scenario = random.Random(seed).choice(phase.scenario_types)
    cfg = {
        **base_cfg,
        'scenario_type': scenario,
        'max_active_hosts': phase.max_active_hosts,
        'topology_churn_rate': phase.topology_churn_rate,
        'topology_migration_rate': phase.topology_migration_rate,
        'topology_arrival_rate': phase.topology_arrival_rate,
    }
    if phase.dhcp_interval > 0:
        cfg['dhcp_interval'] = phase.dhcp_interval
    return NetForgeRLEnv(cfg)


class CurriculumWrapper:
    """Wraps NetForgeRLEnv with automatic phase progression."""

    def __init__(
        self,
        phases: List[PhaseConfig] = PHASES,
        base_cfg: Optional[dict] = None,
        start_phase: int = 0,
        on_phase_advance=None,
    ):
        self.phases = phases
        self.base_cfg = base_cfg or {}
        self._phase_idx = start_phase
        self._on_phase_advance = on_phase_advance
        self._window: Deque[float] = deque(maxlen=phases[start_phase].advance_window)
        self._episode_reward: float = 0.0
        self._env: NetForgeRLEnv = _build_env(self.phase, self.base_cfg, seed=None)

    @property
    def phase(self) -> PhaseConfig:
        return self.phases[self._phase_idx]

    @property
    def phase_index(self) -> int:
        return self._phase_idx

    def _maybe_advance(self) -> bool:
        p = self.phase
        if p.advance_threshold is None:
            return False
        self._window = deque(self._window, maxlen=p.advance_window)
        self._window.append(self._episode_reward)
        if (
            len(self._window) >= p.advance_window
            and np.mean(self._window) >= p.advance_threshold
            and self._phase_idx + 1 < len(self.phases)
        ):
            self._phase_idx += 1
            self._window = deque(maxlen=self.phases[self._phase_idx].advance_window)
            if self._on_phase_advance:
                self._on_phase_advance(self._phase_idx, self.phase.name)
            return True
        return False

    def _curriculum_info(self) -> dict:
        p = self.phase
        progress = len(self._window) / p.advance_window if p.advance_threshold else 1.0
        mean_rew = float(np.mean(self._window)) if self._window else 0.0
        return {
            'phase': p.name,
            'phase_index': self._phase_idx,
            'advance_threshold': p.advance_threshold,
            'mean_reward': mean_rew,
            'window_fill': progress,
        }

    @property
    def agents(self):
        return self._env.agents

    @property
    def possible_agents(self):
        return self._env.possible_agents

    @property
    def observation_spaces(self):
        return self._env.observation_spaces

    @property
    def action_spaces(self):
        return self._env.action_spaces

    def observation_space(self, agent):
        return self._env.observation_space(agent)

    def action_space(self, agent):
        return self._env.action_space(agent)

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        self._episode_reward = 0.0
        self._env = _build_env(self.phase, self.base_cfg, seed=seed)
        obs, info = self._env.reset(seed=seed, options=options)
        for agent in info:
            info[agent]['__curriculum__'] = self._curriculum_info()
        return obs, info

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        obs, rewards, term, trunc, infos = self._env.step(actions)

        step_reward = sum(rewards.values()) * self.phase.reward_scale
        self._episode_reward += step_reward

        advanced = False
        if all(term.values()) or all(trunc.values()):
            advanced = self._maybe_advance()

        curriculum_info = self._curriculum_info()
        curriculum_info['phase_advanced'] = advanced
        for agent in infos:
            infos[agent]['__curriculum__'] = curriculum_info

        if self.phase.reward_scale != 1.0:
            rewards = {a: r * self.phase.reward_scale for a, r in rewards.items()}

        return obs, rewards, term, trunc, infos

    def render(self, mode='rgb_array'):
        return self._env.render(mode)

    def close(self):
        pass
