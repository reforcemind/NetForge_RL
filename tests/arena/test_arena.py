import json

import pytest

from netforge_rl.arena.evaluate import run_episode
from netforge_rl.arena.metrics import cvar, metrics_from_env
from netforge_rl.arena.spec import ARENA_2026
from netforge_rl.baselines.policies import HeuristicBluePolicy, KillChainRedPolicy
from netforge_rl.cli import main
from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.scenarios.yaml_dsl import list_packs, resolve_scenario


@pytest.mark.fast
def test_arena_spec_has_hidden_split():
    spec = ARENA_2026.to_dict()
    assert spec['splits']['hidden']['public'] is False
    assert spec['splits']['dev']['public'] is True
    assert 'mission_success' in spec['metrics']


@pytest.mark.fast
def test_cvar_is_lower_tail():
    assert cvar([0.0, 1.0, 2.0, 3.0, 100.0], alpha=0.2) <= 1.0


@pytest.mark.integration
def test_run_episode_records_competition_metrics():
    rec = run_episode(
        KillChainRedPolicy(seed=0),
        HeuristicBluePolicy(seed=0),
        scenario='ransomware',
        seed=0,
        max_ticks=15,
        graph_obs=False,
    )
    m = rec.metrics
    assert 0.0 <= m.sla_uptime <= 1.0
    assert m.compromised_hosts >= 0
    assert m.false_positives >= 0
    assert m.catastrophic_failure in (0.0, 1.0)


@pytest.mark.fast
def test_community_packs_resolve():
    packs = list_packs()
    assert 'hospital_ransomware' in packs
    spec = resolve_scenario('hospital_ransomware')
    assert spec.base == 'ransomware'
    assert spec.topology_spec is not None


@pytest.mark.integration
def test_pack_builds_env():
    env = NetForgeRLEnv({'scenario_type': 'hospital_ransomware', 'max_ticks': 5})
    env.reset(seed=0)
    names = {h.hostname for h in env._active_hosts()}
    assert 'patient_portal' in names


@pytest.mark.fast
def test_cli_version(capsys):
    assert main(['version']) == 0
    out = capsys.readouterr().out
    assert 'Arena' in out


@pytest.mark.fast
def test_cli_arena_json(capsys):
    assert main(['arena', '--json']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['version'] == '2026'


@pytest.mark.fast
def test_info_exposes_catastrophic_and_fp():
    env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 3})
    env.reset(seed=0)
    _, _, _, _, infos = env.step({a: env.action_space(a).sample() for a in env.agents})
    blue = next(v for k, v in infos.items() if 'blue' in k)
    assert 'false_positives_total' in blue
    assert 'catastrophic_failure' in blue
    assert 'mission_success' in blue
    metrics_from_env(env, infos, 0.0, 0.0)


@pytest.mark.fast
def test_fixed_step_does_not_skip_ticks():
    env = NetForgeRLEnv(
        {'scenario_type': 'ransomware', 'max_ticks': 8, 'time_mode': 'fixed'}
    )
    env.reset(seed=0)
    env.step({a: env.action_space(a).sample() for a in env.agents})
    assert env.current_tick == 1
