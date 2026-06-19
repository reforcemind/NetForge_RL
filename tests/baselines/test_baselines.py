import pytest
pytest.importorskip("PIL")
import json
import numpy as np
import pytest
from netforge_rl.baselines import (
    HeuristicBluePolicy,
    HeuristicRedPolicy,
    RandomPolicy,
    evaluate,
)


@pytest.mark.fast
def test_random_policy_shape(env_sim) -> None:
    a = RandomPolicy(seed=0).act(env_sim, 'blue_dmz')
    assert a.shape == (2,) and a.dtype == np.int64


@pytest.mark.fast
def test_heuristic_blue_targets_compromised_host(env_sim) -> None:
    target = next(
        (ip for ip in env_sim.global_state.all_hosts if not ip.startswith('169.254.'))
    )
    env_sim.global_state.all_hosts[target].compromised_by = 'red_operator'
    a = HeuristicBluePolicy(seed=0).act(env_sim, 'blue_dmz')
    target_ips = sorted(env_sim.global_state.all_hosts.keys())
    assert a[0] == 0 and target_ips[int(a[1])] == target


@pytest.mark.fast
def test_heuristic_red_picks_uncompromised_real_host(env_sim) -> None:
    a = HeuristicRedPolicy(seed=0).act(env_sim, 'red_operator')
    target_ips = sorted(env_sim.global_state.all_hosts.keys())
    chosen = target_ips[int(a[1])]
    assert not chosen.startswith('169.254.')
    assert env_sim.global_state.all_hosts[chosen].compromised_by == 'None'


@pytest.mark.integration
def test_evaluate_runs_and_returns_results() -> None:
    results = evaluate(
        HeuristicBluePolicy(seed=0), scenario='ransomware', episodes=2, max_steps=10
    )
    assert len(results) == 2
    for r in results:
        assert r.model_id == 'heuristic-blue'
        assert r.scenario == 'ransomware'
        assert r.steps <= 10
        assert 'blue_dmz' in r.rewards


@pytest.mark.integration
def test_evaluate_emits_leaderboard_json(tmp_path) -> None:
    from netforge_rl.semantic import append_result, summarize

    path = tmp_path / 'baselines.json'
    for r in evaluate(RandomPolicy(seed=1), episodes=2, max_steps=8):
        append_result(path, r)
    rows = json.loads(path.read_text())
    assert len(rows) == 2
    summary = summarize(path)
    assert summary[0]['model_id'] == 'random'
    assert summary[0]['n_episodes'] == 2
