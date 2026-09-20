from __future__ import annotations

from collections.abc import Callable

from pettingzoo.utils.wrappers import BaseParallelWrapper


def _sla_only(agent, raw, info) -> float:
    if 'blue' not in agent:
        return float(raw)
    return float(info.get('SLA_Uptime_Percentage', 0.0))


def _security_only(agent, raw, info) -> float:
    if 'blue' not in agent:
        return float(raw)
    return float(info.get('security', 0.0)) - 0.1 * float(
        info.get('compromised_hosts', 0.0)
    )


def _fp_penalized(agent, raw, info) -> float:
    if 'blue' not in agent:
        return float(raw)
    return float(raw) - 5.0 * float(info.get('false_positives_total', 0.0))


def _constrained_sla(agent, raw, info) -> float:
    if 'blue' not in agent:
        return float(raw)
    sla = float(info.get('SLA_Uptime_Percentage', 1.0))
    penalty = 0.0 if sla >= 0.9 else 10.0 * (0.9 - sla)
    return float(raw) - penalty


def _safety_critical(agent, raw, info) -> float:
    if 'blue' not in agent:
        return float(raw)
    if float(info.get('catastrophic_failure', 0.0)) >= 1.0:
        return -100.0
    return float(raw)


def _sparse(agent, raw, info) -> float:
    if 'blue' not in agent:
        return 0.0
    return float(info.get('mission_success', 0.0))


VARIANTS: dict[str, Callable] = {
    'default': lambda agent, raw, info: float(raw),
    'sla_only': _sla_only,
    'security_only': _security_only,
    'fp_penalized': _fp_penalized,
    'constrained_sla': _constrained_sla,
    'safety_critical': _safety_critical,
    'sparse': _sparse,
}


class RewardDesignWrapper(BaseParallelWrapper):
    """Swap Blue's reward without editing scenario classes."""

    def __init__(self, env, variant: str = 'default'):
        super().__init__(env)
        if variant not in VARIANTS:
            raise KeyError(
                f'Unknown reward variant {variant!r}. Choose {sorted(VARIANTS)}'
            )
        self.variant = variant
        self._reshape = VARIANTS[variant]

    def step(self, actions):
        obs, rewards, term, trunc, infos = super().step(actions)
        reshaped = {
            agent: self._reshape(agent, rewards[agent], infos.get(agent, {}))
            for agent in rewards
        }
        for agent, info in infos.items():
            info['raw_reward'] = float(rewards.get(agent, 0.0))
            info['reward_variant'] = self.variant
        if self.variant == 'sparse' and (any(term.values()) or any(trunc.values())):
            reshaped = {
                agent: self._reshape(agent, rewards[agent], infos.get(agent, {}))
                for agent in rewards
            }
        return obs, reshaped, term, trunc, infos
