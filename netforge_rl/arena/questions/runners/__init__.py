from netforge_rl.arena.questions.runners.belief import run_belief_vs_oracle
from netforge_rl.arena.questions.runners.clock import run_ood_topology, run_time_mode
from netforge_rl.arena.questions.runners.population import run_red_population
from netforge_rl.arena.questions.runners.probes import run_constrained, run_probe
from netforge_rl.arena.questions.runners.reward import run_reward_design
from netforge_rl.arena.questions.runners.risk import run_risk_sensitive

RUNNERS = {
    'belief-vs-oracle': run_belief_vs_oracle,
    'reward-design': run_reward_design,
    'red-population': run_red_population,
    'risk-sensitive': run_risk_sensitive,
    'time-mode': run_time_mode,
    'ood-topology': run_ood_topology,
    'adaptation': lambda **kw: run_probe('adaptation', **kw),
    'deception': lambda **kw: run_probe('deception', **kw),
    'constrained-sla': run_constrained,
}
