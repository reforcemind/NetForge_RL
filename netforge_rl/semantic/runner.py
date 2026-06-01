"""Zero-shot episode runner — drives a NetForgeRLEnv with LLM-controlled agents.

One :class:`LLMClient` is assigned per controlled agent. Uncontrolled
agents take a no-op (action 0, target 0). On unparseable LLM output the
episode advances with a no-op for that agent and the failure is counted
in :class:`EpisodeResult.invalid_replies`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from netforge_rl.semantic.clients.base import LLMClient
from netforge_rl.semantic.la_wrapper import state_to_text
from netforge_rl.semantic.parser import parse_action
from netforge_rl.semantic.vla_wrapper import build_vla_prompt


@dataclass
class EpisodeResult:
    model_id: str
    scenario: str
    steps: int
    rewards: dict[str, float]
    invalid_replies: dict[str, int]
    final_compromised: int
    final_isolated: int
    metrics: dict[str, float] = field(default_factory=dict)


def _no_op(agent: str) -> np.ndarray:
    return np.array([0, 0], dtype=np.int64)


def run_episode(
    env,
    clients: dict[str, LLMClient],
    *,
    scenario: str = 'ransomware',
    max_steps: int | None = None,
    use_vision: bool = False,
    seed: int = 0,
) -> EpisodeResult:
    """Roll one episode with ``clients[agent_id]`` controlling each agent."""
    env.reset(seed=seed)
    target_ips = sorted(env.global_state.all_hosts.keys())

    cum_reward: dict[str, float] = {a: 0.0 for a in env.possible_agents}
    invalid: dict[str, int] = {a: 0 for a in env.possible_agents}
    step_cap = max_steps if max_steps is not None else env.max_ticks
    steps = 0

    while env.agents and steps < step_cap:
        snap = env.to_envstate()
        actions: dict[str, np.ndarray] = {}
        for agent in env.agents:
            client = clients.get(agent)
            if client is None:
                actions[agent] = _no_op(agent)
                continue
            text = state_to_text(snap, agent, max_hosts=8)
            if use_vision:
                rgb = env.render(mode='rgb_array')
                prompt: dict[str, Any] = build_vla_prompt(rgb, text)
            else:
                prompt = {'text': text}
            prompt['_agent_id'] = agent
            prompt['_target_ips'] = target_ips

            parsed = parse_action(client.act(prompt), agent, target_ips)
            if parsed is None:
                invalid[agent] += 1
                actions[agent] = _no_op(agent)
            else:
                actions[agent] = np.array(parsed, dtype=np.int64)

        _, rewards, term, trunc, _ = env.step(actions)
        for a, r in rewards.items():
            cum_reward[a] = cum_reward.get(a, 0.0) + float(r)
        steps += 1
        if all(term.values()) or all(trunc.values()):
            break

    final_compromised = sum(
        1 for h in env.global_state.all_hosts.values() if h.compromised_by != 'None'
    )
    final_isolated = sum(
        1 for h in env.global_state.all_hosts.values() if h.status == 'isolated'
    )

    primary_id = next(iter(clients.values())).model_id if clients else 'none'
    return EpisodeResult(
        model_id=primary_id,
        scenario=scenario,
        steps=steps,
        rewards=cum_reward,
        invalid_replies=invalid,
        final_compromised=final_compromised,
        final_isolated=final_isolated,
    )
