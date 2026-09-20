from __future__ import annotations

from collections.abc import Mapping

from pettingzoo.utils.wrappers import BaseParallelWrapper

DEFAULT_CONSTRAINTS: dict[str, dict] = {
    'sla_uptime': {'min': 0.90},
    'false_positives_total': {'max': 2.0},
    'disruption': {'max': 0.40},
    'catastrophic_failure': {'max': 0.0},
}


class ConstrainedEnvWrapper(BaseParallelWrapper):
    """Lagrangian costs from Blue info (SLA, FPs, disruption, kinetic)."""

    def __init__(
        self,
        env,
        constraints: Mapping[str, dict] | None = None,
        lambda_cost: float = 0.0,
        team: str = 'blue',
    ):
        super().__init__(env)
        self.constraints = dict(constraints or DEFAULT_CONSTRAINTS)
        self.lambda_cost = float(lambda_cost)
        self.team = team

    def reset(self, seed=None, options=None):
        obs, infos = super().reset(seed=seed, options=options)
        return obs, self._annotate(infos, {})

    def step(self, actions):
        obs, rewards, term, trunc, infos = super().step(actions)
        costs = self._costs(infos)
        if self.lambda_cost:
            rewards = {
                agent: float(rewards[agent])
                - (
                    self.lambda_cost * costs.get(agent, 0.0)
                    if self.team in agent
                    else 0.0
                )
                for agent in rewards
            }
        return obs, rewards, term, trunc, self._annotate(infos, costs)

    def _costs(self, infos: dict) -> dict[str, float]:
        costs = {}
        for agent, info in infos.items():
            cost = 0.0
            violations = {}
            for key, spec in self.constraints.items():
                value = float(info.get(key, 0.0))
                if 'min' in spec and value < spec['min']:
                    gap = spec['min'] - value
                    violations[key] = gap
                    cost += gap
                if 'max' in spec and value > spec['max']:
                    gap = value - spec['max']
                    violations[key] = gap
                    cost += gap
            info['constraint_violations'] = violations
            costs[agent] = cost
        return costs

    def _annotate(self, infos: dict, costs: dict) -> dict:
        for agent, info in infos.items():
            info['constraint_cost'] = float(costs.get(agent, 0.0))
        return infos
