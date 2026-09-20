from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from netforge_rl.arena.evaluate import _act
from netforge_rl.environment.presets import make_env


@dataclass
class Trajectory:
    scenario: str
    seed: int
    policy: str
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    terminals: list = field(default_factory=list)
    truncations: list = field(default_factory=list)
    infos: list = field(default_factory=list)
    timeouts: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rewards)


def collect_episode(
    red_policy,
    blue_policy,
    *,
    scenario: str = 'ransomware',
    seed: int = 0,
    max_ticks: int = 100,
    controlled: str = 'blue',
    quality: str = 'mixed',
) -> Trajectory:
    """Roll one trajectory for offline RL. Stores the controlled team's steps."""
    env = make_env('medium', scenario_type=scenario, seed=None, max_ticks=max_ticks)
    obs, _ = env.reset(seed=seed)
    traj = Trajectory(scenario=scenario, seed=seed, policy=quality)
    while env.agents:
        actions = {}
        for agent_id in env.agents:
            policy = red_policy if 'red' in agent_id else blue_policy
            actions[agent_id] = _act(policy, env, agent_id, obs.get(agent_id, {}))
        next_obs, rewards, term, trunc, infos = env.step(actions)
        team_agents = [a for a in rewards if controlled in a]
        step_obs = {a: _serialize_obs(obs[a]) for a in team_agents if a in obs}
        step_act = {a: np.asarray(actions[a]).tolist() for a in team_agents}
        step_rew = {a: float(rewards[a]) for a in team_agents}
        traj.observations.append(step_obs)
        traj.actions.append(step_act)
        traj.rewards.append(step_rew)
        traj.terminals.append({a: bool(term.get(a, False)) for a in team_agents})
        traj.truncations.append({a: bool(trunc.get(a, False)) for a in team_agents})
        traj.infos.append(
            {
                a: {
                    k: v
                    for k, v in infos.get(a, {}).items()
                    if isinstance(v, (int, float, str, bool))
                }
                for a in team_agents
            }
        )
        obs = next_obs
        if all(term.values()) or all(trunc.values()):
            break
    return traj


def collect_dataset(
    red_factory,
    blue_factory,
    *,
    n_episodes: int = 4,
    scenario: str = 'ransomware',
    max_ticks: int = 40,
    quality: str = 'mixed',
    seed0: int = 0,
) -> list[Trajectory]:
    return [
        collect_episode(
            red_factory(),
            blue_factory(),
            scenario=scenario,
            seed=seed0 + i,
            max_ticks=max_ticks,
            quality=quality,
        )
        for i in range(n_episodes)
    ]


def _serialize_obs(obs: dict) -> dict:
    out = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            out[key] = value.astype(np.float32).tolist()
        else:
            out[key] = value
    return out
