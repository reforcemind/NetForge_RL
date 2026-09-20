from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from netforge_rl.arena.evaluate import run_episode
from netforge_rl.arena.metrics import cvar, worst_case
from netforge_rl.arena.questions.policies import blue, killchain


def run_risk_sensitive(*, seeds: Sequence[int], max_ticks: int) -> dict:
    returns = []
    catastrophic = []
    sla = []
    for seed in seeds:
        rec = run_episode(
            killchain(),
            blue(),
            scenario='ransomware',
            seed=int(seed),
            max_ticks=max_ticks,
        )
        returns.append(rec.metrics.blue_return)
        catastrophic.append(rec.metrics.catastrophic_failure)
        sla.append(rec.metrics.sla_uptime)
    return {
        'n_seeds': len(returns),
        'mean_blue_return': round(float(np.mean(returns)), 4),
        'cvar05': round(cvar(returns, 0.05), 4),
        'worst_case': round(worst_case(returns), 4),
        'mean_sla': round(float(np.mean(sla)), 4),
        'catastrophic_rate': round(float(np.mean(catastrophic)), 4),
    }
