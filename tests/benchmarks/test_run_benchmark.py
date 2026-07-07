import pytest

pytest.importorskip('PIL')

from benchmarks.competition_eval import PolicyAgent, run_episode
from benchmarks.run_benchmark import make_agent, run_benchmark
from netforge_rl.baselines.policies import HeuristicBluePolicy


@pytest.mark.integration
def test_run_benchmark_executes_episodes(tmp_path, monkeypatch):
    """The default path must actually run episodes, not silently no-op."""
    import benchmarks.run_benchmark as rb

    monkeypatch.setattr(rb, 'RESULTS_DIR', tmp_path)
    result = run_benchmark(
        name='smoke',
        team='blue',
        red_agent=make_agent('heuristic', 'red'),
        blue_agent=make_agent('heuristic', 'blue'),
        scenarios=['ransomware'],
        n_seeds=1,
        max_ticks=20,
    )
    assert result['total_episodes'] == 1
    assert result['overall']['n_episodes'] == 1


@pytest.mark.integration
def test_policy_agent_binds_env_and_acts():
    """PolicyAgent must drive the real heuristic, not fall back to random."""
    agent = PolicyAgent(HeuristicBluePolicy(seed=0))
    ep = run_episode(agent, agent, scenario='ransomware', seed=0, max_ticks=10)
    assert ep.blue_agent_errors == 0
    assert ep.steps > 0
