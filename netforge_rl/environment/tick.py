"""One tick: enqueue, clock, resolve, SIEM, physics, topology, observe."""

from __future__ import annotations

import numpy as np

from netforge_rl.core.action import ActionEffect, BaseAction
from netforge_rl.core.observation import BaseObservation
from netforge_rl.core.registry import action_registry
from netforge_rl.environment.constants import (
    BLUE_INFLIGHT_CAP,
    MAX_ACTION_DURATION,
    OBS_VECTOR_DIM,
    SIEM_BUFFER_CAP,
    SIEM_ENCODE_N,
)
from netforge_rl.environment.events import PendingAction, as_pending
from netforge_rl.nlp.log_encoder import EMBEDDING_DIM
from netforge_rl.siem.event_templates import sysmon_1


def _inflight_counts(env) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in env.event_queue:
        counts[event.agent] = counts.get(event.agent, 0) + 1
    return counts


def enqueue_actions(env, agent_actions: dict) -> None:
    inflight = _inflight_counts(env)
    for agent, action_int in agent_actions.items():
        if env.current_tick < env.global_state.agent_locked_until.get(agent, 0):
            continue
        if isinstance(action_int, BaseAction):
            action = action_int
        else:
            env.ordered_hosts = sorted(env.global_state.all_hosts.keys())
            action = action_registry.instantiate_action(
                agent, action_int, env.ordered_hosts
            )
            if action is None:
                continue
        if 'blue' in agent.lower():
            if inflight.get(agent, 0) >= BLUE_INFLIGHT_CAP:
                continue
            inflight[agent] = inflight.get(agent, 0) + 1
        if env.global_state.agent_energy.get(agent, 0) < action.cost:
            continue
        if not action.validate(env.global_state):
            continue
        env.global_state.agent_energy[agent] -= action.cost
        eta = getattr(action, 'duration', 1)
        completion_tick = env.current_tick + eta
        effect = action.execute(env.global_state)
        effect.action = action
        env.global_state.agent_locked_until[agent] = completion_tick
        env.event_queue.append(
            PendingAction(
                completion_tick=completion_tick,
                agent=agent,
                action=action,
                effect=effect,
                target_ip=getattr(action, 'target_ip', None),
                start_tick=env.current_tick,
            )
        )


def cancel_red_on_isolate(env) -> None:
    for event in list(env.event_queue):
        if type(event.action).__name__ != 'IsolateHost':
            continue
        if event.completion_tick > env.current_tick:
            continue
        target = event.target_ip
        for red_event in list(env.event_queue):
            if 'red' not in red_event.agent.lower():
                continue
            if red_event.target_ip != target:
                continue
            if red_event in env.event_queue:
                env.event_queue.remove(red_event)
            env.global_state.agent_locked_until[red_event.agent] = env.current_tick


def advance_clock(env) -> tuple[float, float]:
    prev = env.current_tick
    if env.time_mode == 'fixed':
        env.current_tick += 1
    elif env.event_queue:
        nxt = min(e.completion_tick for e in env.event_queue)
        env.current_tick = max(env.current_tick + 1, nxt)
    else:
        env.current_tick += 1
    delta_t = float(env.current_tick - prev)
    return delta_t, delta_t / MAX_ACTION_DURATION


def pop_due_effects(env) -> tuple[dict[str, ActionEffect], dict[str, dict]]:
    intended: dict[str, ActionEffect] = {}
    metadata: dict[str, dict] = {}
    remaining = []
    for event in env.event_queue:
        if env.current_tick >= event.completion_tick:
            intended[event.agent] = event.effect
            metadata[event.agent] = {
                'name': type(event.action).__name__,
                'target_ip': event.target_ip,
            }
        else:
            remaining.append(event)
    env.event_queue = remaining
    return intended, metadata


def emit_resolved_siem(env, resolved: dict, metadata: dict) -> None:
    for agent, effect in resolved.items():
        meta = metadata.get(agent, {})
        env.siem_logger.log_action(
            action_name=meta.get('name', 'UnknownAction'),
            effect=effect,
            global_state=env.global_state,
            agent_id=agent,
            target_ip=effect.observation_data.get('exploit'),
        )
    for agent, effect in resolved.items():
        if 'red' not in agent or not effect.success:
            continue
        target_ip = effect.observation_data.get('exploit', 'unknown')
        host = env.global_state.all_hosts.get(target_ip)
        subnet = host.subnet_cidr if host else 'unknown'
        env.siem_logger._push_to_buffer(
            sysmon_1(agent, process='exploit_payload', rng=env.siem_logger._rng),
            subnet,
            env.global_state,
        )
        if host and getattr(host, 'contains_honeytokens', False):
            env.siem_logger._push_to_buffer(
                {
                    'signature': 'HONEYTOKEN_TRIGGERED',
                    'target': target_ip,
                    'agent': agent,
                    'severity': 10,
                },
                subnet,
                env.global_state,
            )


def drop_stale_events(env) -> None:
    valid = set(env.global_state.all_hosts.keys())
    env.event_queue = [
        e for e in env.event_queue if e.target_ip is None or e.target_ip in valid
    ]
    env._cached_action_masks = {agent: env.action_mask(agent) for agent in env.agents}


def maybe_dhcp(env) -> None:
    interval = env.dhcp_interval
    if interval > 0 and env.current_tick % interval == 0:
        env.global_state.reallocate_dhcp(rng=env._py_random)
        drop_stale_events(env)


def tick_physics(env) -> None:
    alerts, deltas = env.physics_engine.tick(env.global_state)
    for key, val in deltas:
        env.global_state.apply_delta(key, val)
    ot_subnet = '10.0.99.0/24'
    for alert in alerts:
        env.siem_logger._push_to_buffer(alert, ot_subnet, env.global_state)


def tick_correlator(env) -> None:
    for log, subnet in env.correlator.correlate(env.global_state):
        env.global_state.siem_log_buffer.append((log, subnet))
        if len(env.global_state.siem_log_buffer) > SIEM_BUFFER_CAP:
            env.global_state.siem_log_buffer.pop(0)


def tick_topology(env) -> None:
    events = env.topology_engine.tick(env.global_state)
    if not events:
        return
    drop_stale_events(env)
    for ev in events:
        env.siem_logger._push_to_buffer(
            {
                'signature': f'TOPOLOGY_{ev.kind.upper()}',
                'detail': ev.detail,
                'severity': 3,
            },
            ev.detail.get('subnet', ev.detail.get('new_subnet', 'unknown')),
            env.global_state,
        )


def encode_siem(env) -> dict[str, np.ndarray]:
    vecs = {}
    for agent in env.agents:
        if 'blue' not in agent.lower():
            continue
        subnet_tag = agent.split('_')[1] if '_' in agent else 'dmz'
        logs = env.siem_logger.get_filtered_logs(
            env.global_state, subnet_tag=subnet_tag, n=SIEM_ENCODE_N
        )
        vecs[agent] = env.log_encoder.encode_buffer(logs, agg='mean')
    return vecs


def collect_observations(
    env,
    resolved: dict,
    metadata: dict,
    delta_t_norm: float,
    siem_vecs: dict,
) -> tuple[dict, dict]:
    pcap = (
        env.pcap_synthesizer.synthesize(
            env.global_state, env.current_tick, env.max_ticks
        )
        if env.pcap_synthesizer
        else None
    )
    nodes = (
        env.pcap_synthesizer.node_features(env.global_state)
        if env.pcap_synthesizer
        else None
    )
    blue_comm = env._build_blue_comm()
    observations = {}
    rewards = {}
    for agent in env.agents:
        obs = BaseObservation(agent)
        obs.update_from_state(env.global_state, resolved)
        if 'blue' in agent.lower():
            siem = siem_vecs.get(agent, np.zeros(EMBEDDING_DIM, dtype=np.float32))
        else:
            siem = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        agent_obs = {
            'obs': obs.to_numpy(max_size=OBS_VECTOR_DIM),
            'action_mask': env._cached_action_masks[agent],
            'siem_embedding': siem,
            'adj_matrix': env._get_adj_matrix_for(agent).flatten(),
            'delta_t': np.array([delta_t_norm], dtype=np.float32),
        }
        if 'blue' in agent.lower():
            agent_obs['blue_comm'] = blue_comm
        if env.pcap_obs:
            agent_obs['pcap'] = pcap
            agent_obs['node_features'] = nodes
        observations[agent] = agent_obs
        effect = resolved.get(agent)
        rewards[agent] = env.scenario.calculate_reward(agent, env.global_state, effect)
    if env.trajectory_recorder is not None:
        for agent, effect in resolved.items():
            meta = metadata.get(agent, {})
            env.trajectory_recorder.record_step(
                tick=env.current_tick,
                agent_id=agent,
                action_name=meta.get('name', 'UnknownAction'),
                target_ip=meta.get('target_ip'),
                success=effect.success,
                reward=float(rewards.get(agent, 0.0)),
            )
    return observations, rewards


def run_step(env, agent_actions: dict):
    env.event_queue = [as_pending(event) for event in env.event_queue]
    enqueue_actions(env, agent_actions)
    cancel_red_on_isolate(env)
    delta_t, delta_t_norm = advance_clock(env)
    env.global_state.current_tick = env.current_tick
    env.global_state.subnet_bandwidth.clear()

    noise = env.green_agent.generate_noise(
        env.current_tick, env.global_state, rng=env._py_random
    )
    for anomaly in noise.get('alerts', []):
        env.siem_logger._push_to_buffer(
            anomaly['data'], anomaly['subnet'], env.global_state
        )

    intended, metadata = pop_due_effects(env)
    resolved = env.resolution_engine.resolve(intended)
    env._apply_state_deltas(resolved)
    env._update_episode_metrics(resolved)
    emit_resolved_siem(env, resolved, metadata)
    env.siem_logger.log_background_noise(env.global_state)
    maybe_dhcp(env)
    tick_physics(env)
    tick_correlator(env)
    tick_topology(env)

    terminate = env.scenario.check_termination(env.global_state)
    truncated = env.current_tick >= env.max_ticks
    truncate = {agent: truncated for agent in env.agents}
    env.siem_logger.release(env.global_state)
    observations, rewards = collect_observations(
        env, resolved, metadata, delta_t_norm, encode_siem(env)
    )
    env.agents = [
        agent for agent in env.agents if not terminate[agent] and not truncate[agent]
    ]
    infos = env._extract_agent_infos(observations, resolved, rewards)
    for agent in env.agents:
        if agent in infos:
            infos[agent]['delta_t'] = delta_t
            infos[agent]['delta_t_norm'] = delta_t_norm
    return observations, rewards, terminate, truncate, infos
