from typing import Dict, Tuple
import numpy as np
from netforge_rl.scenarios import get_scenario_class

from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.observation import BaseObservation
from netforge_rl.core.registry import action_registry, team_of
import netforge_rl.actions  # noqa: F401 — registers all action decorators

from netforge_rl.core.physics import ConflictResolutionEngine
from netforge_rl.environment.base_env import BaseNetForgeRLEnv
from netforge_rl.topologies.network_generator import NetworkGenerator
from netforge_rl.topologies.dynamic_topology import TopologyEventEngine
from netforge_rl.scenarios.ot_physics import PLCPhysicsEngine
from netforge_rl.agents.green_agent import GreenAgent
from netforge_rl.docker_bridge.bridge import DockerBridge
from netforge_rl.siem.siem_logger import SIEMLogger
from netforge_rl.siem.pcap_synthesizer import (
    PcapSynthesizer,
    N_PACKETS,
    PACKET_DIM,
    NODE_DIM,
)
from netforge_rl.siem.correlator import SIEMCorrelator
from netforge_rl.nlp.log_encoder import LogEncoder, EMBEDDING_DIM
from netforge_rl.siem.event_templates import sysmon_1, seed_events
from netforge_rl.core.functional import from_global_state
from netforge_rl.environment.constants import (  # noqa: F401 — re-exported
    MAX_ACTION_DURATION,
    PADDING_SUBNET,
)
from netforge_rl.environment.metrics import EpisodeMetricsMixin
from netforge_rl.environment.observations import ObservationMixin
from netforge_rl.environment.spaces import build_action_spaces, build_observation_spaces
import random


class NetForgeRLEnv(BaseNetForgeRLEnv, EpisodeMetricsMixin, ObservationMixin):
    """PettingZoo-style MARL environment for the NetForge cybersecurity sim."""

    metadata = {'render_modes': ['ansi', 'rgb_array'], 'name': 'netforge_rl_v3'}

    def __init__(self, scenario_config: dict):
        cfg = scenario_config or {}
        self.network_generator = NetworkGenerator(
            config_path=cfg.get('topology_path'),
            max_active_hosts=cfg.get('max_active_hosts'),
            evaluation_mode=cfg.get('evaluation_mode', False),
        )
        self.log_latency = cfg.get('log_latency', 0)
        self.dhcp_interval = cfg.get('dhcp_interval', 40)
        self.record_siem = cfg.get('record_siem', False)
        self.green_agent = GreenAgent()
        self.possible_agents = [
            'red_operator',
            'blue_dmz',
            'blue_internal',
            'blue_restricted',
        ]
        self.agents = self.possible_agents[:]
        scenario_cls = get_scenario_class(cfg.get('scenario_type', 'ransomware'))
        self.scenario = scenario_cls(self.agents)

        self.global_state = self.network_generator.generate()
        self.resolution_engine = ConflictResolutionEngine()

        self.docker_bridge = DockerBridge(mode=cfg.get('docker_mode', 'sim'))
        self.global_state.docker_bridge = self.docker_bridge

        self.siem_logger = SIEMLogger()
        self.log_encoder = LogEncoder(backend=cfg.get('nlp_backend', 'tfidf'))
        self.topology_engine = TopologyEventEngine(
            churn_rate=cfg.get('topology_churn_rate', 0.0),
            migration_rate=cfg.get('topology_migration_rate', 0.0),
            arrival_rate=cfg.get('topology_arrival_rate', 0.0),
        )
        self.physics_engine = PLCPhysicsEngine()
        self.correlator = SIEMCorrelator()
        self.pcap_obs = cfg.get('pcap_obs', False)
        self.pcap_synthesizer = PcapSynthesizer() if self.pcap_obs else None
        if cfg.get('record_trajectory', False):
            from netforge_rl.render.trajectory import TrajectoryRecorder

            self.trajectory_recorder = TrajectoryRecorder()
        else:
            self.trajectory_recorder = None

        self.observation_spaces = build_observation_spaces(
            self.possible_agents, self.pcap_obs
        )
        self.action_spaces = build_action_spaces(self.possible_agents)
        self.max_ticks = cfg.get('max_ticks', 1000)
        self.current_tick = 0
        self.event_queue = []

    def reset(
        self, seed=None, options=None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """Reset the environment."""
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
        self.global_state.agent_energy = {agent: 50 for agent in self.agents}
        self.global_state.agent_funds = {
            agent: 10000 if 'blue' in agent else 5000 for agent in self.agents
        }
        self.global_state.agent_compute = {agent: 1000 for agent in self.agents}
        self.global_state.business_downtime_score = 0.0
        # SIEM log buffer and research metrics
        self.global_state.siem_log_buffer = []
        self.episode_metrics = {
            'infection_times': {},  # IP -> tick
            'detection_times': {},  # IP -> tick (first SIEM alert)
            'isolation_times': {},  # IP -> tick
            'exfiltrated_data': 0.0,
            'sla_uptime_sum': 0.0,
            'steps_count': 0,
            'deception_hits': 0,  # red actions that struck a decoy/honeytoken
            'red_actions': 0,  # resolved red actions, for efficacy ratio
            'attack_techniques': set(),  # MITRE ATT&CK technique ids exercised
        }

        observations = {}
        for agent_id in self.agents:
            obs = BaseObservation(agent_id)
            obs.update_from_state(self.global_state, [])
            agent_obs = {
                'obs': obs.to_numpy(max_size=256),
                'action_mask': self._cached_action_masks[agent_id],
                'siem_embedding': np.zeros(EMBEDDING_DIM, dtype=np.float32),
                'adj_matrix': self.global_state.get_adjacency_matrix().flatten(),
                'delta_t': np.zeros(1, dtype=np.float32),
            }
            if 'blue' in agent_id:
                agent_obs['blue_comm'] = np.zeros(100, dtype=np.float32)
            if self.pcap_obs:
                agent_obs['pcap'] = np.zeros((N_PACKETS, PACKET_DIM), dtype=np.float32)
                agent_obs['node_features'] = np.zeros((100, NODE_DIM), dtype=np.float32)
            observations[agent_id] = agent_obs
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

    def action_mask(self, agent: str) -> np.ndarray:
        """Generate a binary action mask of shape (132,) = (32 + 100)."""
        mask = np.zeros(132, dtype=np.int8)
        for action_id in action_registry._actions.get(team_of(agent), {}):
            if action_id < 32:
                mask[action_id] = 1
        ordered = sorted(self.global_state.all_hosts.keys())
        for i, ip in enumerate(ordered[:100]):
            host = self.global_state.all_hosts.get(ip)
            if host and host.status != 'isolated':
                mask[32 + i] = 1
        return mask

    def step(
        self, agent_actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, BaseObservation],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """Process actions and advance simulation."""

        per_agent_inflight: Dict[str, int] = {}
        for event in self.event_queue:
            per_agent_inflight[event['agent']] = (
                per_agent_inflight.get(event['agent'], 0) + 1
            )

        for agent, action_int in agent_actions.items():
            if self.current_tick < self.global_state.agent_locked_until.get(agent, 0):
                continue

            if isinstance(action_int, BaseAction):
                action = action_int
            else:
                self.ordered_hosts = sorted(self.global_state.all_hosts.keys())
                action = action_registry.instantiate_action(
                    agent, action_int, self.ordered_hosts
                )
                if action is None:
                    continue

            # Cap Blue agents at 2 in-flight actions.
            if 'blue' in agent.lower():
                if per_agent_inflight.get(agent, 0) >= 2:
                    continue
                per_agent_inflight[agent] = per_agent_inflight.get(agent, 0) + 1

            if self.global_state.agent_energy.get(agent, 0) < action.cost:
                continue
            if not action.validate(self.global_state):
                continue

            self.global_state.agent_energy[agent] -= action.cost

            eta = getattr(action, 'duration', 1)
            completion_tick = self.current_tick + eta
            effect = action.execute(self.global_state)
            effect.action = action

            self.global_state.agent_locked_until[agent] = completion_tick
            self.event_queue.append(
                {
                    'completion_tick': completion_tick,
                    'agent': agent,
                    'action': action,
                    'effect': effect,
                    'target_ip': getattr(action, 'target_ip', None),
                    'start_tick': self.current_tick,
                }
            )

        for event in list(self.event_queue):
            if (
                type(event['action']).__name__ == 'IsolateHost'
                and event['completion_tick'] <= self.current_tick
            ):
                target_to_isolate = event['target_ip']
                for red_event in list(self.event_queue):
                    if (
                        'red' in red_event['agent'].lower()
                        and red_event['target_ip'] == target_to_isolate
                    ):
                        if red_event in self.event_queue:
                            self.event_queue.remove(red_event)
                        self.global_state.agent_locked_until[red_event['agent']] = (
                            self.current_tick
                        )

        prev_tick = self.current_tick
        if self.event_queue:
            next_event_tick = min(e['completion_tick'] for e in self.event_queue)
            self.current_tick = max(self.current_tick + 1, next_event_tick)
        else:
            self.current_tick += 1

        delta_t = float(self.current_tick - prev_tick)
        delta_t_norm = delta_t / MAX_ACTION_DURATION

        self.global_state.current_tick = self.current_tick
        self.global_state.subnet_bandwidth.clear()

        noise_data = self.green_agent.generate_noise(
            self.current_tick, self.global_state, rng=self._py_random
        )
        for anomaly in noise_data.get('alerts', []):
            self.siem_logger._push_to_buffer(
                anomaly['data'], anomaly['subnet'], self.global_state
            )

        intended_effects = {}
        action_metadata = {}
        remaining_events = []
        for event in self.event_queue:
            if self.current_tick >= event['completion_tick']:
                agent = event['agent']
                intended_effects[agent] = event['effect']
                action_metadata[agent] = {
                    'name': type(event['action']).__name__,
                    'target_ip': event.get('target_ip'),
                }
            else:
                remaining_events.append(event)
        self.event_queue = remaining_events

        resolved_effects = self.resolution_engine.resolve(intended_effects)
        self._apply_state_deltas(resolved_effects)

        self._update_episode_metrics(resolved_effects)

        for res_agent, res_effect in resolved_effects.items():
            meta = action_metadata.get(res_agent, {})
            self.siem_logger.log_action(
                action_name=meta.get('name', 'UnknownAction'),
                effect=res_effect,
                global_state=self.global_state,
                agent_id=res_agent,
                target_ip=res_effect.observation_data.get('exploit'),
            )

        for res_agent, res_effect in resolved_effects.items():
            if 'red' not in res_agent or not res_effect.success:
                continue
            target_ip = res_effect.observation_data.get('exploit', 'unknown')
            host = self.global_state.all_hosts.get(target_ip)
            subnet = host.subnet_cidr if host else 'unknown'

            self.siem_logger._push_to_buffer(
                sysmon_1(
                    res_agent, process='exploit_payload', rng=self.siem_logger._rng
                ),
                subnet,
                self.global_state,
            )
            if host and getattr(host, 'contains_honeytokens', False):
                self.siem_logger._push_to_buffer(
                    {
                        'signature': 'HONEYTOKEN_TRIGGERED',
                        'target': target_ip,
                        'agent': res_agent,
                        'severity': 10,
                    },
                    subnet,
                    self.global_state,
                )

        self.siem_logger.log_background_noise(self.global_state)

        if self.dhcp_interval > 0 and self.current_tick % self.dhcp_interval == 0:
            self.global_state.reallocate_dhcp(rng=self._py_random)
            valid_ips = set(self.global_state.all_hosts.keys())
            self.event_queue = [
                e
                for e in self.event_queue
                if e.get('target_ip') is None or e['target_ip'] in valid_ips
            ]
            self._cached_action_masks = {
                agent: self.action_mask(agent) for agent in self.agents
            }

        physics_alerts, physics_deltas = self.physics_engine.tick(self.global_state)
        for delta_key, delta_val in physics_deltas:
            self.global_state.apply_delta(delta_key, delta_val)
        ot_subnet = '10.0.99.0/24'
        for alert in physics_alerts:
            self.siem_logger._push_to_buffer(alert, ot_subnet, self.global_state)

        for incident_log, incident_subnet in self.correlator.correlate(
            self.global_state
        ):
            self.global_state.siem_log_buffer.append((incident_log, incident_subnet))
            if len(self.global_state.siem_log_buffer) > 64:
                self.global_state.siem_log_buffer.pop(0)

        topo_events = self.topology_engine.tick(self.global_state)
        if topo_events:
            valid_ips = set(self.global_state.all_hosts.keys())
            self.event_queue = [
                e
                for e in self.event_queue
                if e.get('target_ip') is None or e['target_ip'] in valid_ips
            ]
            for ev in topo_events:
                self.siem_logger._push_to_buffer(
                    {
                        'signature': f'TOPOLOGY_{ev.kind.upper()}',
                        'detail': ev.detail,
                        'severity': 3,
                    },
                    ev.detail.get('subnet', ev.detail.get('new_subnet', 'unknown')),
                    self.global_state,
                )
            self._cached_action_masks = {
                agent: self.action_mask(agent) for agent in self.agents
            }

        observations = {}
        rewards = {}
        terminate = self.scenario.check_termination(self.global_state)
        is_truncated = self.current_tick >= self.max_ticks
        truncate = {agent: is_truncated for agent in self.agents}

        self.siem_logger.release(self.global_state)

        # Encode SIEM logs.
        agent_siem_vecs = {}
        for agent in self.agents:
            if 'blue' in agent.lower():
                subnet_tag = agent.split('_')[1] if '_' in agent else 'dmz'
                subset_logs = self.siem_logger.get_filtered_logs(
                    self.global_state, subnet_tag=subnet_tag, n=8
                )
                agent_siem_vecs[agent] = self.log_encoder.encode_buffer(
                    subset_logs, agg='mean'
                )

        pcap_snapshot = (
            self.pcap_synthesizer.synthesize(
                self.global_state, self.current_tick, self.max_ticks
            )
            if self.pcap_synthesizer
            else None
        )
        node_feat_snapshot = (
            self.pcap_synthesizer.node_features(self.global_state)
            if self.pcap_synthesizer
            else None
        )

        blue_comm = self._build_blue_comm()

        for agent in self.agents:
            obs = BaseObservation(agent)
            obs.update_from_state(self.global_state, resolved_effects)

            obs_array = obs.to_numpy(max_size=256)

            if 'blue' in agent.lower():
                agent_siem_vec = agent_siem_vecs.get(
                    agent, np.zeros(EMBEDDING_DIM, dtype=np.float32)
                )
            else:
                agent_siem_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

            agent_obs = {
                'obs': obs_array,
                'action_mask': self._cached_action_masks[agent],
                'siem_embedding': agent_siem_vec,
                'adj_matrix': self._get_adj_matrix_for(agent).flatten(),
                'delta_t': np.array([delta_t_norm], dtype=np.float32),
            }
            if 'blue' in agent.lower():
                agent_obs['blue_comm'] = blue_comm
            if self.pcap_obs:
                agent_obs['pcap'] = pcap_snapshot
                agent_obs['node_features'] = node_feat_snapshot
            observations[agent] = agent_obs
            agent_effect = resolved_effects.get(agent)
            rewards[agent] = self.scenario.calculate_reward(
                agent, self.global_state, agent_effect
            )

        if self.trajectory_recorder is not None:
            for agent, effect in resolved_effects.items():
                meta = action_metadata.get(agent, {})
                self.trajectory_recorder.record_step(
                    tick=self.current_tick,
                    agent_id=agent,
                    action_name=meta.get('name', 'UnknownAction'),
                    target_ip=meta.get('target_ip'),
                    success=effect.success,
                    reward=float(rewards.get(agent, 0.0)),
                )

        self.agents = [
            agent
            for agent in self.agents
            if not terminate[agent] and not truncate[agent]
        ]

        infos = self._extract_agent_infos(observations, resolved_effects, rewards)

        for agent in self.agents:
            if agent in infos:
                infos[agent]['delta_t'] = delta_t
                infos[agent]['delta_t_norm'] = delta_t_norm

        return observations, rewards, terminate, truncate, infos

    def render(self, mode: str = 'rgb_array'):
        """Render the environment frame."""
        if mode == 'ansi':
            return None
        if mode != 'rgb_array':
            raise ValueError(f'Unsupported render mode: {mode}')
        from netforge_rl.render import render_rgb, snapshot_from_envstate

        return render_rgb(snapshot_from_envstate(self.to_envstate()))

    def to_envstate(self):
        """Return a frozen EnvState PyTree snapshot."""
        return from_global_state(self.global_state, tuple(self.possible_agents))

    def _apply_state_deltas(self, effects: Dict[str, ActionEffect]):
        """Apply state deltas to global_state."""
        for effect in effects.values():
            if not effect.success:
                continue
            if isinstance(effect.state_deltas, dict):
                for delta_key, delta_val in effect.state_deltas.items():
                    self.global_state.apply_delta(delta_key, delta_val)
            elif isinstance(effect.state_deltas, list):
                for delta_cmd in effect.state_deltas:
                    self.global_state.apply_delta(delta_cmd)
