"""Three standard scoreboard modes for the Zero-Shot Incident Response leaderboard.

* :func:`zero_shot_defender` — LLM controls Blue; a scripted Red attacks.
* :func:`zero_shot_attacker` — LLM controls Red; a heuristic SOC defends.
* :func:`fm_vs_fm`           — one LLM per side. Reports each side's score.
"""

import numpy as np

from netforge_rl.semantic.la_wrapper import state_to_text
from netforge_rl.semantic.parser import parse_action


BLUE_AGENTS = ('blue_dmz', 'blue_internal', 'blue_restricted')
RED_AGENTS = ('red_operator',)


def _no_op():
    return np.array([0, 0], dtype=np.int64)


def _llm_action(env, agent, client, target_ips):
    """Query ``client`` for one action; ``None`` if the reply is unparseable."""
    snap = env.to_envstate()
    prompt = {
        'text': state_to_text(snap, agent, max_hosts=8),
        '_agent_id': agent,
        '_target_ips': target_ips,
    }
    parsed = parse_action(client.act(prompt), agent, target_ips)
    if parsed is None:
        return None
    return np.array(parsed, dtype=np.int64)


def _drive(env, role_clients, role_fallback, *, max_steps, seed):
    """Common drive loop.

    ``role_clients`` maps agent_id -> LLMClient. Agents not in ``role_clients``
    fall back to ``role_fallback(env, agent_id)``.
    """
    env.reset(seed=seed)
    target_ips = sorted(env.global_state.all_hosts.keys())

    cum_reward = {a: 0.0 for a in env.possible_agents}
    invalid = {a: 0 for a in env.possible_agents}
    steps = 0
    step_cap = max_steps if max_steps is not None else env.max_ticks

    while env.agents and steps < step_cap:
        actions = {}
        for agent in env.agents:
            client = role_clients.get(agent)
            if client is not None:
                act = _llm_action(env, agent, client, target_ips)
                if act is None:
                    invalid[agent] += 1
                    act = _no_op()
                actions[agent] = act
            else:
                actions[agent] = role_fallback(env, agent)
        _, rewards, term, trunc, _ = env.step(actions)
        for a, r in rewards.items():
            cum_reward[a] += float(r)
        steps += 1
        if all(term.values()) or all(trunc.values()):
            break

    return cum_reward, invalid, steps


def zero_shot_defender(env, blue_client, *, max_steps=None, seed=0):
    """LLM controls all Blue agents; Red follows the heuristic-red baseline."""
    from netforge_rl.baselines.policies import HeuristicRedPolicy

    blue_clients = {a: blue_client for a in BLUE_AGENTS if a in env.possible_agents}
    red_baseline = HeuristicRedPolicy(seed=seed).act
    cum_reward, invalid, steps = _drive(
        env, blue_clients, red_baseline, max_steps=max_steps, seed=seed,
    )
    return {
        'mode': 'zero_shot_defender',
        'model_id': blue_client.model_id,
        'side': 'blue',
        'cum_reward': cum_reward,
        'invalid_replies': invalid,
        'steps': steps,
        **_score_extras(env),
    }


def zero_shot_attacker(env, red_client, *, max_steps=None, seed=0):
    """LLM controls Red; Blue follows the heuristic-blue baseline."""
    from netforge_rl.baselines.policies import HeuristicBluePolicy

    red_clients = {a: red_client for a in RED_AGENTS if a in env.possible_agents}
    blue_baseline = HeuristicBluePolicy(seed=seed).act
    cum_reward, invalid, steps = _drive(
        env, red_clients, blue_baseline, max_steps=max_steps, seed=seed,
    )
    return {
        'mode': 'zero_shot_attacker',
        'model_id': red_client.model_id,
        'side': 'red',
        'cum_reward': cum_reward,
        'invalid_replies': invalid,
        'steps': steps,
        **_score_extras(env),
    }


def fm_vs_fm(env, blue_client, red_client, *, max_steps=None, seed=0):
    """LLM-vs-LLM: one client per side. Both sides scored against each other."""
    clients = {a: blue_client for a in BLUE_AGENTS if a in env.possible_agents}
    clients.update({a: red_client for a in RED_AGENTS if a in env.possible_agents})

    def _never(env, agent):
        return _no_op()

    cum_reward, invalid, steps = _drive(
        env, clients, _never, max_steps=max_steps, seed=seed,
    )
    return {
        'mode': 'fm_vs_fm',
        'blue_model_id': blue_client.model_id,
        'red_model_id': red_client.model_id,
        'cum_reward': cum_reward,
        'invalid_replies': invalid,
        'steps': steps,
        **_score_extras(env),
    }


def _score_extras(env):
    return {
        'final_compromised': sum(
            1 for h in env.global_state.all_hosts.values()
            if h.compromised_by != 'None'
        ),
        'final_isolated': sum(
            1 for h in env.global_state.all_hosts.values()
            if h.status == 'isolated'
        ),
    }
