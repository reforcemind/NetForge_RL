import random
from typing import ClassVar

import numpy as np

import netforge_rl.actions  # noqa: F401 — registers all action decorators
from netforge_rl.agents.green_agent import GreenAgent
from netforge_rl.core.action import ActionEffect
from netforge_rl.core.functional import from_global_state
from netforge_rl.core.physics import ConflictResolutionEngine
from netforge_rl.core.registry import action_registry, team_of
from netforge_rl.docker_bridge.bridge import DockerBridge
from netforge_rl.environment.base_env import BaseNetForgeRLEnv
from netforge_rl.environment.config import EnvConfig
from netforge_rl.environment.constants import (  # noqa: F401 — re-exported
    ACTION_MASK_DIM,
    MAX_ACTION_DURATION,
    N_ACTION_TYPES,
    N_HOST_SLOTS,
    PADDING_SUBNET,
)
from netforge_rl.environment.metrics import EpisodeMetricsMixin
from netforge_rl.environment.observations import ObservationMixin
from netforge_rl.environment.reset import (
    empty_episode_metrics,
    initial_observations,
    seed_agent_budgets,
)
from netforge_rl.environment.spaces import build_action_spaces, build_observation_spaces
from netforge_rl.environment.tick import run_step
from netforge_rl.nlp.log_encoder import LogEncoder
from netforge_rl.scenarios import get_scenario_class
from netforge_rl.scenarios.ot_physics import PLCPhysicsEngine
from netforge_rl.siem.correlator import SIEMCorrelator
from netforge_rl.siem.event_templates import seed_events
from netforge_rl.siem.pcap_synthesizer import PcapSynthesizer
from netforge_rl.siem.siem_logger import SIEMLogger
from netforge_rl.topologies.dynamic_topology import TopologyEventEngine
from netforge_rl.topologies.network_generator import NetworkGenerator


class NetForgeRLEnv(BaseNetForgeRLEnv, EpisodeMetricsMixin, ObservationMixin):
    """PettingZoo parallel env."""

    metadata: ClassVar[dict] = {
        'render_modes': ['ansi', 'rgb_array'],
        'name': 'netforge_rl_v4',
    }

    def __init__(self, scenario_config: dict | EnvConfig | None = None):
        from netforge_rl.scenarios.yaml_dsl import expand_scenario_config

        raw = (
            scenario_config.to_dict()
            if isinstance(scenario_config, EnvConfig)
            else dict(scenario_config or {})
        )
        self.config = EnvConfig.from_mapping(expand_scenario_config(raw))
        cfg = self.config
        self.network_generator = NetworkGenerator(
            config_path=cfg.topology_path,
            max_active_hosts=cfg.max_active_hosts,
            evaluation_mode=cfg.evaluation_mode,
            topology_spec=cfg.topology_spec,
        )
        self.log_latency = cfg.log_latency
        self.dhcp_interval = cfg.dhcp_interval
        self.record_siem = cfg.record_siem
        self.pcap_obs = cfg.pcap_obs
        self.max_ticks = cfg.max_ticks
        self.time_mode = cfg.time_mode
        self.green_agent = GreenAgent()
        self.possible_agents = list(cfg.agents.all())
        self.agents = self.possible_agents[:]
        scenario_cls = get_scenario_class(cfg.scenario_type)
        self.scenario = scenario_cls(self.agents)
        self.global_state = self.network_generator.generate()
        self.resolution_engine = ConflictResolutionEngine()
        self.docker_bridge = DockerBridge(mode=cfg.docker_mode)
        self.global_state.docker_bridge = self.docker_bridge
        self.siem_logger = SIEMLogger()
        self.log_encoder = LogEncoder(backend=cfg.nlp_backend)
        self.topology_engine = TopologyEventEngine(
            churn_rate=cfg.topology.churn,
            migration_rate=cfg.topology.migration,
            arrival_rate=cfg.topology.arrival,
        )
        self.physics_engine = PLCPhysicsEngine()
        self.correlator = SIEMCorrelator()
        self.pcap_synthesizer = PcapSynthesizer() if cfg.pcap_obs else None
        if cfg.record_trajectory:
            from netforge_rl.render.trajectory import TrajectoryRecorder

            self.trajectory_recorder = TrajectoryRecorder()
        else:
            self.trajectory_recorder = None
        self.observation_spaces = build_observation_spaces(
            self.possible_agents, cfg.pcap_obs
        )
        self.action_spaces = build_action_spaces(self.possible_agents)
        self.current_tick = 0
        self.event_queue = []

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        self.np_random = np.random.default_rng(seed)
        self._py_random = random.Random(seed)
        seed_events(seed)
        self.siem_logger = SIEMLogger(
            seed=seed, latency=self.log_latency, capture=self.record_siem
        )
        self.docker_bridge.teardown_all()
        self.docker_bridge.reseed(seed)
        self.global_state = self.network_generator.generate(seed=seed)
        self.global_state.docker_bridge = self.docker_bridge
        self.agents = self.possible_agents[:]
        self.ordered_hosts = sorted(self.global_state.all_hosts.keys())
        self._cached_action_masks = {
            agent: self.action_mask(agent) for agent in self.agents
        }
        seed_agent_budgets(self)
        self.episode_metrics = empty_episode_metrics()
        observations = initial_observations(self)
        self.current_tick = 0
        self.event_queue = []
        self.topology_engine.reset(seed=seed)
        self.physics_engine.reset(seed=seed)
        self.correlator.reset()
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.reset(
                scenario=self.scenario.__class__.__name__, seed=seed or 0
            )
        if self.pcap_synthesizer:
            self.pcap_synthesizer.reset(seed=seed)
        return observations, {agent: {} for agent in self.agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def action_mask(self, agent: str):
        mask = np.zeros(ACTION_MASK_DIM, dtype=np.int8)
        for action_id in action_registry._actions.get(team_of(agent), {}):
            if action_id < N_ACTION_TYPES:
                mask[action_id] = 1
        ordered = sorted(self.global_state.all_hosts.keys())
        for i, ip in enumerate(ordered[:N_HOST_SLOTS]):
            host = self.global_state.all_hosts.get(ip)
            if host and host.status != 'isolated':
                mask[N_ACTION_TYPES + i] = 1
        return mask

    def step(self, agent_actions: dict):
        return run_step(self, agent_actions)

    def render(self, mode: str = 'rgb_array'):
        if mode == 'ansi':
            return None
        if mode != 'rgb_array':
            raise ValueError(f'Unsupported render mode: {mode}')
        from netforge_rl.render import render_rgb, snapshot_from_envstate

        return render_rgb(snapshot_from_envstate(self.to_envstate()))

    def to_envstate(self):
        return from_global_state(self.global_state, tuple(self.possible_agents))

    def _apply_state_deltas(self, effects: dict[str, ActionEffect]):
        for effect in effects.values():
            if not effect.success:
                continue
            if isinstance(effect.state_deltas, dict):
                for key, val in effect.state_deltas.items():
                    self.global_state.apply_delta(key, val)
            elif isinstance(effect.state_deltas, list):
                for cmd in effect.state_deltas:
                    self.global_state.apply_delta(cmd)
