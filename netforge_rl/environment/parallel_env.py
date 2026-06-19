from typing import Dict, Tuple
import numpy as np
import gymnasium as gym
from netforge_rl.scenarios import get_scenario_class
from netforge_rl.scenarios.apt_espionage import AptEspionageScenario

from netforge_rl.core.action import BaseAction, ActionEffect
from netforge_rl.core.observation import BaseObservation
from netforge_rl.core.registry import action_registry
import netforge_rl.actions
from netforge_rl.core.physics import ConflictResolutionEngine
from netforge_rl.environment.base_env import BaseNetForgeRLEnv
from netforge_rl.topologies.network_generator import NetworkGenerator
from netforge_rl.agents.green_agent import GreenAgent
from netforge_rl.docker_bridge.bridge import DockerBridge
from netforge_rl.siem.siem_logger import SIEMLogger
from netforge_rl.nlp.log_encoder import LogEncoder, EMBEDDING_DIM


# Normalization constant for Neural ODE integration
MAX_ACTION_DURATION = 50.0


class NetForgeRLEnv(BaseNetForgeRLEnv):
    """PettingZoo-style MARL environment for the NetForge cybersecurity sim."""

    metadata = {'render_modes': ['ansi', 'rgb_array'], 'name': 'netforge_rl_v3'}

    def __init__(self, scenario_config: dict):
        cfg = scenario_config or {}
        self.network_generator = NetworkGenerator(config_path=cfg.get('topology_path'))
        self.log_latency = cfg.get('log_latency', 2)
        self.green_agent = GreenAgent()
        self.possible_agents = [
            'red_operator',
            'blue_dmz',
            'blue_internal',
            'blue_restricted',
        ]
        self.agents = self.possible_agents[:]
        try:
            scenario_cls = get_scenario_class(cfg.get('scenario_type', 'ransomware'))
        except KeyError:
            scenario_cls = AptEspionageScenario
        self.scenario = scenario_cls(self.agents)

        self.global_state = self.network_generator.generate()
        self.resolution_engine = ConflictResolutionEngine()

        self.docker_bridge = DockerBridge(mode=cfg.get('docker_mode', 'sim'))
        self.global_state.docker_bridge = self.docker_bridge

        self.siem_logger = SIEMLogger()
        self.log_encoder = LogEncoder(backend=cfg.get('nlp_backend', 'tfidf'))

        self.observation_spaces = {
            agent: gym.spaces.Dict(
                {
                    'obs': gym.spaces.Box(
                        low=-1.0, high=1.0, shape=(256,), dtype=np.float32
                    ),
                    'action_mask': gym.spaces.Box(
                        low=0, high=1, shape=(32 + 100,), dtype=np.int8
                    ),
                    'siem_embedding': gym.spaces.Box(
                        low=-1.0, high=1.0, shape=(EMBEDDING_DIM,), dtype=np.float32
                    ),
                    'adj_matrix': gym.spaces.Box(
                        low=0.0, high=1.0, shape=(10000,), dtype=np.float32
                    ),
                    'delta_t': gym.spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                }
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: gym.spaces.MultiDiscrete(
                [32, 100]
            )  # [Action Type (max 32), Target IP Index (max 100 padded)]
            for agent in self.possible_agents
        }
        self.max_ticks = cfg.get('max_ticks', 1000)
        self.current_tick = 0
        self.event_queue = []

    def reset(
        self, seed=None, options=None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """Reset the environment."""
        if seed is not None:
            import random as _random

            _random.seed(seed)
            np.random.seed(seed)
        self.docker_bridge.teardown_all()
        self.global_state = self.network_generator.generate(seed=seed)
        self.global_state.docker_bridge = self.docker_bridge
        self.agents = self.possible_agents[:]
        self.ordered_hosts = sorted(self.global_state.all_hosts.keys())
        self._cached_action_masks = {agent: self.action_mask(agent) for agent in self.agents}
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
        }

        observations = {}
        for agent_id in self.agents:
            obs = BaseObservation(agent_id)
            obs.update_from_state(self.global_state, [])
            observations[agent_id] = {
                'obs': obs.to_numpy(max_size=256),
                'action_mask': self._cached_action_masks[agent_id],
                'siem_embedding': np.zeros(EMBEDDING_DIM, dtype=np.float32),
                'adj_matrix': self.global_state.get_adjacency_matrix().flatten(),
                'delta_t': np.zeros(1, dtype=np.float32),
            }
        self.current_tick = 0
        self.event_queue = []

        return observations, {agent: {} for agent in self.agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def action_mask(self, agent: str) -> np.ndarray:
        """Generate a binary action mask (132,)."""
        mask = np.zeros(132, dtype=np.int8)
        valid_action_types = 17 if 'red' in agent.lower() else 15
        mask[:valid_action_types] = 1
        mask[32 : 32 + 100] = 1
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
            self.global_state.agent_energy[agent] -= action.cost

            if action.validate(self.global_state):
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
            self.current_tick, self.global_state
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

        from netforge_rl.siem.event_templates import sysmon_1

        for res_agent, res_effect in resolved_effects.items():
            if 'red' not in res_agent or not res_effect.success:
                continue
            target_ip = res_effect.observation_data.get('exploit', 'unknown')
            host = self.global_state.all_hosts.get(target_ip)
            subnet = host.subnet_cidr if host else 'unknown'

            self.siem_logger._push_to_buffer(
                sysmon_1(res_agent, process='exploit_payload'),
                subnet,
                self.global_state,
            )
            if host and getattr(host, 'contains_honeytokens', False):
                self.siem_logger._push_to_buffer(
                    {
                        'signature': 'HONEYTOKEN_TRIGGERED',
                        'severity': 10,
                        'target': target_ip,
                        'agent': res_agent,
                    },
                    subnet,
                    self.global_state,
                )

        self.siem_logger.log_background_noise(self.global_state)

        observations = {}
        rewards = {}
        terminate = self.scenario.check_termination(self.global_state)

        if self.current_tick % 40 == 0:
            self.global_state.reallocate_dhcp()

        is_truncated = self.current_tick >= self.max_ticks
        truncate = {agent: is_truncated for agent in self.agents}

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

            observations[agent] = {
                'obs': obs_array,
                'action_mask': self._cached_action_masks[agent],
                'siem_embedding': agent_siem_vec,
                'adj_matrix': self.global_state.get_adjacency_matrix().flatten(),
                'delta_t': np.array([delta_t_norm], dtype=np.float32),
            }
            agent_effect = resolved_effects.get(agent)
            rewards[agent] = self.scenario.calculate_reward(
                agent, self.global_state, agent_effect
            )

        self.agents = [
            agent
            for agent in self.agents
            if not terminate[agent] and not truncate[agent]
        ]


        infos = self._extract_agent_infos(observations, resolved_effects)


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
        from netforge_rl.core.functional import from_global_state

        return from_global_state(self.global_state, tuple(self.possible_agents))

    def _update_episode_metrics(self, resolved_effects):
        """Update episode security metrics."""
        for agent, effect in resolved_effects.items():
            if not effect.success or not isinstance(effect.state_deltas, dict):
                continue
            for delta_key, delta_val in effect.state_deltas.items():
                parts = delta_key.split('/')
                if len(parts) != 3 or parts[0] != 'hosts':
                    continue
                ip, attribute = parts[1], parts[2]
                if attribute == 'privilege' and delta_val in ('User', 'Root'):
                    self.episode_metrics['infection_times'].setdefault(
                        ip, self.current_tick
                    )
                elif attribute == 'status' and delta_val == 'isolated':
                    self.episode_metrics['isolation_times'].setdefault(
                        ip, self.current_tick
                    )

        total = max(len(self.global_state.all_hosts), 1)
        healthy = sum(
            1 for h in self.global_state.all_hosts.values()
            if h.compromised_by == 'None' and h.status == 'online'
        )
        self.episode_metrics['sla_uptime_sum'] += healthy / total
        self.episode_metrics['steps_count'] += 1

    def _apply_state_deltas(self, effects: Dict[str, ActionEffect]):
        """Apply state deltas to global_state."""
        for agent_id, effect in effects.items():
            if not effect.success:
                continue
            if isinstance(effect.state_deltas, dict):
                for delta_key, delta_val in effect.state_deltas.items():
                    self.global_state.apply_delta(delta_key, delta_val)
            elif isinstance(effect.state_deltas, list):
                for delta_cmd in effect.state_deltas:
                    self.global_state.apply_delta(delta_cmd)

    def _extract_agent_infos(self, observations: dict, resolved_effects: dict) -> dict:
        """Extract per-agent metrics info dict."""
        infos = {}
        for agent in observations:
            agent_effect = resolved_effects.get(agent)
            info: dict = {}

            false_positives = 0
            successful_exploits = 0
            hosts_isolated = 0
            services_restored = 0

            if (
                agent_effect
                and agent_effect.success
                and isinstance(agent_effect.state_deltas, dict)
            ):
                for delta_key, delta_val in agent_effect.state_deltas.items():
                    if 'status' in delta_key and delta_val == 'isolated':
                        hosts_isolated += 1
                        parts = delta_key.split('/')
                        if len(parts) >= 2:
                            host = self.global_state.all_hosts.get(parts[1])
                            if host and host.compromised_by == 'None':
                                false_positives += 1
                    elif 'privilege' in delta_key and delta_val in ('User', 'Root'):
                        successful_exploits += 1
                    elif 'status' in delta_key and delta_val == 'online':
                        services_restored += 1

            info['false_positives'] = float(false_positives)
            info['successful_exploits'] = float(successful_exploits)
            info['hosts_isolated'] = float(hosts_isolated)
            info['services_restored'] = float(services_restored)

            target_ip = (
                getattr(agent_effect.action, 'target_ip', None)
                if agent_effect else None
            )
            self.ordered_hosts = sorted(self.global_state.all_hosts.keys())
            info['target_ip_index'] = (
                self.ordered_hosts.index(target_ip)
                if target_ip and target_ip in self.global_state.all_hosts
                else None
            )

            info['agent_energy'] = float(self.global_state.agent_energy.get(agent, 0))
            info['compromised_hosts'] = float(
                sum(
                    1
                    for h in self.global_state.all_hosts.values()
                    if h.compromised_by != 'None'
                )
            )
            info['isolated_hosts'] = float(
                sum(
                    1
                    for h in self.global_state.all_hosts.values()
                    if h.status == 'isolated'
                )
            )

            sla_final = (
                self.episode_metrics['sla_uptime_sum']
                / self.episode_metrics['steps_count']
                if self.episode_metrics['steps_count'] > 0
                else 1.0
            )
            info['SLA_Uptime_Percentage'] = float(sla_final)

            # Mean Time To Containment
            mttc_vals = []
            for ip, t_iso in self.episode_metrics['isolation_times'].items():
                if ip in self.episode_metrics['infection_times']:
                    mttc_vals.append(
                        t_iso - self.episode_metrics['infection_times'][ip]
                    )
            info['MTTC'] = float(sum(mttc_vals) / len(mttc_vals)) if mttc_vals else 0.0


            info['Total_Exfiltrated_Data'] = float(
                self.episode_metrics['exfiltrated_data']
            )

            infos[agent] = info

        return infos

    def global_state_vector(self) -> np.ndarray:
        """Generate a flat 512-dim global state vector."""
        priv_codes = {'None': 0.0, 'User': 0.5, 'Root': 1.0}

        vec = []
        for ip in self.ordered_hosts[:100]:
            host = self.global_state.all_hosts[ip]
            vec.extend([
                priv_codes.get(host.privilege, 0.0),
                1.0 if host.status == 'online' else 0.0,
                1.0 if host.decoy != 'inactive' else 0.0,
            ])
        vec.append(self.global_state.business_downtime_score / 100.0)
        vec.append(float(self.current_tick) / float(self.max_ticks))
        for agent in self.possible_agents:
            vec.append(float(self.global_state.agent_energy.get(agent, 0)) / 100.0)

        result = np.zeros(512, dtype=np.float32)
        v_arr = np.array(vec, dtype=np.float32)
        result[: len(v_arr)] = v_arr
        return result
