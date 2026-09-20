from netforge_rl.arena.questions.types import ResearchQuestion

QUESTIONS: tuple[ResearchQuestion, ...] = (
    ResearchQuestion(
        id='belief-vs-oracle',
        title='Belief graph vs privileged oracle',
        question=(
            'How much of the true compromise state is missing from Blue SIEM '
            'belief graph, and does an oracle leak change a telemetry-only policy?'
        ),
        why='POMDP / privileged-information ablation. Oracle graphs are diagnostic-only.',
        protocol='Plant a silent compromise, compare belief vs oracle node masks and labels.',
        metrics=('visible_ratio', 'compromise_recall', 'heuristic_sla'),
        status='runnable',
        note='Train Blue on SIEM belief. Oracle graphs are diagnostics only.',
    ),
    ResearchQuestion(
        id='reward-design',
        title='Does the reward isolate the network?',
        question=(
            'How does swapping Blue reward (SLA-only, FP-penalized, default) '
            'change isolation, false positives, and mission success on the same seeds?'
        ),
        why='A reward that pays for isolation will take the whole net down.',
        protocol='RewardDesignWrapper variants, identical seeds, Arena metrics.',
        metrics=('isolated_hosts', 'false_positives', 'sla_uptime', 'mission_success'),
        status='runnable',
        note='Check isolation and FPs, not only return.',
    ),
    ResearchQuestion(
        id='red-population',
        title='One Blue vs a Red population',
        question=(
            'Does a defender that looks strong vs kill-chain Red still hold SLA '
            'against random and heuristic attackers?'
        ),
        why='A defender that only beats one scripted Red overfits.',
        protocol='evaluate_population with random / heuristic / kill-chain Red.',
        metrics=('mean_sla_across_red', 'worst_sla_across_red'),
        status='runnable',
        note='Score against the worst Red in the pool, not only the training opponent.',
    ),
    ResearchQuestion(
        id='risk-sensitive',
        title='Mean return vs CVaR vs worst case',
        question=(
            'Is a high mean Blue return hiding a heavy lower tail or kinetic failures?'
        ),
        why='Safety-critical defense cannot rank on mean reward.',
        protocol='Several seeds; report mean, CVaR-5%, worst-case, catastrophic rate.',
        metrics=('mean_blue_return', 'cvar05', 'worst_case', 'catastrophic_rate'),
        status='runnable',
        note='CVaR and kinetic failures sit next to mean return.',
    ),
    ResearchQuestion(
        id='time-mode',
        title='Event-driven vs fixed-step time',
        question=(
            'Does the same policy produce different SLA and tick counts under '
            'durative event time versus a fixed tick clock?'
        ),
        why='Exploits and isolates take ticks; the clock is part of the gym.',
        protocol='Identical seed; env_overrides time_mode=event vs fixed.',
        metrics=('steps', 'sla_uptime', 'blue_return'),
        status='runnable',
        note='Compare event-driven time to a fixed tick clock.',
    ),
    ResearchQuestion(
        id='ood-topology',
        title='Train-size vs larger live network',
        question='Does a medium-topology defender keep SLA when max_active_hosts grows?',
        why='Generalization over topology size, not a new host type.',
        protocol='Same policy; default hosts vs max_active_hosts=40.',
        metrics=('sla_uptime', 'security', 'compromised_hosts'),
        status='runnable',
        note='Hold out larger live networks than training.',
    ),
    ResearchQuestion(
        id='adaptation',
        title='Mid-episode attacker shift',
        question=(
            'After Red switches from kill-chain to random spread, does Blue contain '
            'new infections or keep fighting the first strategy?'
        ),
        why='Adaptation is a capability, not a training trick.',
        protocol='AdaptationShift diagnostic probe.',
        metrics=('score', 'new_infections', 'contained_after'),
        status='runnable',
        note='Red can change strategy mid-episode.',
    ),
    ResearchQuestion(
        id='deception',
        title='Honeypot / decoy resistance',
        question='Does Blue waste isolation on decoys, or ignore them and miss real hosts?',
        why='Deception is a Blue capability probe, not more Red exploits.',
        protocol='DeceptionResistance diagnostic.',
        metrics=('score',),
        status='runnable',
        note='Honeypots and decoys can waste Blue isolation.',
    ),
    ResearchQuestion(
        id='constrained-sla',
        title='Defense under SLA / FP constraints',
        question=(
            'Can Blue keep security while respecting SLA >= 0.9 and a false-positive cap?'
        ),
        why='Constrained RL belongs in the gym; Lagrangian/CPO stay outside.',
        protocol='ConstrainedEnvWrapper costs on the same seeds as unconstrained.',
        metrics=('constraint_cost', 'sla_uptime', 'false_positives', 'security'),
        status='runnable',
        note='Keep SLA / FP costs on the same seeds as unconstrained.',
    ),
    ResearchQuestion(
        id='gnn-belief',
        title='GNN on belief graphs (protocol)',
        question='Does GNN-MAPPO trained on SIEM belief transfer to hidden topologies?',
        why='The gym ships belief graphs; you bring the GNN trainer.',
        protocol='Train on infos[blue][graph]; eval Arena hidden; oracle=cheat ablation.',
        metrics=('mission_success', 'ood', 'oracle_gap'),
        status='protocol',
        note='Do not vendor MAPPO into this tree.',
    ),
    ResearchQuestion(
        id='offline-to-online',
        title='Offline SOC logs then online fine-tune (protocol)',
        question='Does pretraining on mixed trajectories beat learning from scratch?',
        why='Offline RL needs frozen datasets, not a bigger simulator.',
        protocol='netforge collect + HDF5/npz; fine-tune; Arena hidden.',
        metrics=('mission_success', 'false_positives', 'cvar05'),
        status='protocol',
        note='Dump mixed trajectories, then fine-tune online.',
    ),
    ResearchQuestion(
        id='league-psro',
        title='Self-play league / PSRO (protocol)',
        question='Does a Blue that best-responds to a Red league beat a single opponent?',
        why='MARL as a game. Payoff matrices ship; PSRO trainers do not.',
        protocol='netforge_rl.league.payoff_matrix; historical checkpoints as the pool.',
        metrics=('worst_red', 'empirical_game'),
        status='protocol',
        note='League trainers live outside this repo.',
    ),
)

BY_ID = {q.id: q for q in QUESTIONS}
