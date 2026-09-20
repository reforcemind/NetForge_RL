from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from netforge_rl.environment.presets import EVAL_SEEDS

ARENA_VERSION = '2026'
ARENA_NAME = f'NetForge Arena {ARENA_VERSION}'

SplitName = Literal['train', 'dev', 'hidden']

PUBLIC_SCENARIOS: tuple[str, ...] = (
    'ransomware',
    'apt_espionage',
    'cloud_hybrid',
    'iot_grid',
    'ot_stuxnet',
)

# Community packs are first-class Arena scenarios once loaded from YAML.
PACK_SCENARIOS: tuple[str, ...] = (
    'hospital_ransomware',
    'cloud_iam',
    'enterprise_apt',
    'finance',
    'iot_fleet',
    'ot_plant',
    'zero_trust',
)

TRAIN_SEEDS: tuple[int, ...] = tuple(range(20))
DEV_SEEDS: tuple[int, ...] = EVAL_SEEDS
# Disjoint reserved block. Official leaderboard organizers may replace this
# with unpublished topology/attacker packs; the public proxy is still OOD.
HIDDEN_SEEDS: tuple[int, ...] = tuple(range(50_000, 50_020))

REFERENCE_BASELINES: tuple[str, ...] = (
    'random',
    'heuristic-blue',
    'heuristic-red',
    'killchain-red',
    'ippo',
    'mappo',
    'rmappo',
    'gnn-mappo',
    'qmix',
    'llm',
)

HEADLINE_CAPABILITIES: tuple[str, ...] = (
    'memory',
    'temporal',
    'precision',
    'safety',
    'generalization',
    'deception',
    'adaptation',
    'attention',
)

COMPETITION_METRICS: tuple[str, ...] = (
    'mission_success',
    'disruption',
    'security',
    'false_positives',
    'exfiltration',
    'compromised_hosts',
    'catastrophic_failure',
    'sla_uptime',
    'cvar_return',
    'worst_case_return',
)


@dataclass(frozen=True)
class Split:
    name: SplitName
    seeds: tuple[int, ...]
    evaluation_mode: bool
    public: bool
    description: str


@dataclass(frozen=True)
class ArenaSpec:
    """Frozen evaluation contract for a yearly Arena release."""

    version: str = ARENA_VERSION
    scenarios: tuple[str, ...] = PUBLIC_SCENARIOS
    difficulty: str = 'medium'
    max_ticks: int = 200
    train: Split = field(
        default_factory=lambda: Split(
            name='train',
            seeds=TRAIN_SEEDS,
            evaluation_mode=False,
            public=True,
            description='Public training topologies.',
        )
    )
    dev: Split = field(
        default_factory=lambda: Split(
            name='dev',
            seeds=DEV_SEEDS,
            evaluation_mode=True,
            public=True,
            description='Held-out topologies for model selection.',
        )
    )
    hidden: Split = field(
        default_factory=lambda: Split(
            name='hidden',
            seeds=HIDDEN_SEEDS,
            evaluation_mode=True,
            public=False,
            description='Organizer split. Local runs use the public proxy.',
        )
    )
    compute_budget_env_steps: int = 1_000_000
    baselines: tuple[str, ...] = REFERENCE_BASELINES

    def split(self, name: SplitName) -> Split:
        return {'train': self.train, 'dev': self.dev, 'hidden': self.hidden}[name]

    def to_dict(self) -> dict:
        return {
            'name': ARENA_NAME,
            'version': self.version,
            'scenarios': list(self.scenarios),
            'difficulty': self.difficulty,
            'max_ticks': self.max_ticks,
            'compute_budget_env_steps': self.compute_budget_env_steps,
            'baselines': list(self.baselines),
            'splits': {
                s.name: {
                    'n_seeds': len(s.seeds),
                    'evaluation_mode': s.evaluation_mode,
                    'public': s.public,
                    'description': s.description,
                }
                for s in (self.train, self.dev, self.hidden)
            },
            'metrics': list(COMPETITION_METRICS),
            'capabilities': list(HEADLINE_CAPABILITIES),
        }


ARENA_2026 = ArenaSpec()


def current_arena() -> ArenaSpec:
    return ARENA_2026
