from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from netforge_rl.environment.presets import make_config

PACKS_DIR = Path(__file__).parent / 'packs'


@dataclass
class ScenarioSpec:
    """Declarative experiment: topology, scenario family, knobs. No simulator edits."""

    name: str
    base: str = 'ransomware'
    pack: str = 'custom'
    difficulty: str = 'medium'
    max_ticks: int = 200
    description: str = ''
    topology_path: str | None = None
    topology_spec: dict | None = None
    reward_variant: str | None = None
    log_latency: int | None = None
    time_mode: str = 'event'
    extras: dict = field(default_factory=dict)

    def env_config(self, **overrides) -> dict:
        cfg = make_config(
            self.difficulty,
            scenario_type=self.base,
            max_ticks=overrides.pop('max_ticks', self.max_ticks),
            **{
                k: v
                for k, v in {
                    'topology_path': self.topology_path,
                    'topology_spec': self.topology_spec,
                    'log_latency': self.log_latency,
                    'time_mode': self.time_mode,
                }.items()
                if v is not None
            },
        )
        cfg.update(self.extras)
        cfg.update(overrides)
        return cfg


def packs_dir() -> Path:
    return PACKS_DIR


def list_packs() -> list[str]:
    names = []
    if PACKS_DIR.exists():
        names.extend(p.stem for p in sorted(PACKS_DIR.glob('*.yaml')))
    return names


def _from_mapping(data: dict, source: str = 'custom') -> ScenarioSpec:
    topo = data.get('topology')
    topology_spec = None
    topology_path = data.get('topology_path')
    if isinstance(topo, dict) and (topo.get('subnets') or topo.get('hosts')):
        topology_spec = topo
    elif isinstance(topo, str):
        topology_path = topo
    extras = dict(data.get('env', {}) or {})
    return ScenarioSpec(
        name=data.get('name', source),
        base=data.get('base', data.get('scenario_type', 'ransomware')),
        pack=data.get('pack', source),
        difficulty=data.get('difficulty', 'medium'),
        max_ticks=int(data.get('max_ticks', 200)),
        description=data.get('description', ''),
        topology_path=topology_path,
        topology_spec=topology_spec,
        reward_variant=data.get('reward_variant'),
        log_latency=data.get('log_latency'),
        time_mode=data.get('time_mode', extras.pop('time_mode', 'event')),
        extras=extras,
    )


def load_scenario_file(path: str | Path) -> ScenarioSpec:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    spec = _from_mapping(data, source=path.stem)
    if spec.topology_path and not Path(spec.topology_path).is_absolute():
        candidate = path.parent / spec.topology_path
        if candidate.exists():
            spec.topology_path = str(candidate)
    return spec


def resolve_scenario(name_or_path: str) -> ScenarioSpec:
    """Accept a builtin name, a community pack stem, or a YAML path."""
    from netforge_rl.scenarios import is_builtin

    raw = str(name_or_path)
    as_path = Path(raw)
    if as_path.suffix in {'.yaml', '.yml'} and as_path.exists():
        return load_scenario_file(as_path)
    pack = PACKS_DIR / f'{raw}.yaml'
    if pack.exists():
        return load_scenario_file(pack)
    if is_builtin(raw):
        return ScenarioSpec(name=raw, base=raw, pack='builtin')
    raise KeyError(
        f'Unknown scenario {raw!r}. Builtin families or packs: {sorted(list_packs())}'
    )


def expand_scenario_config(cfg: dict) -> dict:
    """Resolve pack aliases into a builtin scenario_type plus topology knobs."""
    cfg = dict(cfg or {})
    name = cfg.get('scenario_type', 'ransomware')
    from netforge_rl.scenarios import is_builtin

    if is_builtin(name):
        return cfg
    try:
        spec = resolve_scenario(name)
    except KeyError:
        return cfg
    merged = spec.env_config()
    user = {k: v for k, v in cfg.items() if k != 'scenario_type'}
    merged.update(user)
    merged['scenario_type'] = spec.base
    return merged


def make_env_from_spec(
    spec: ScenarioSpec,
    difficulty: str | None = None,
    seed: int | None = None,
    **overrides,
):
    from netforge_rl.environment.parallel_env import NetForgeRLEnv

    if difficulty:
        spec = replace(spec, difficulty=difficulty)
    env = NetForgeRLEnv(spec.env_config(**overrides))
    if spec.reward_variant:
        from netforge_rl.rewards.variants import RewardDesignWrapper

        env = RewardDesignWrapper(env, variant=spec.reward_variant)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_env_from_yaml(
    path: str | Path,
    seed: int | None = None,
    **overrides,
):
    return make_env_from_spec(load_scenario_file(path), seed=seed, **overrides)
