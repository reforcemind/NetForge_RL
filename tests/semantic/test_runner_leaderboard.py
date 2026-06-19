import pytest
pytest.importorskip("PIL")
import pytest
from netforge_rl.semantic import EpisodeResult, append_result, run_episode, summarize
from netforge_rl.semantic.clients import MockLLMClient


@pytest.mark.integration
def test_run_episode_with_scripted_client(env_sim) -> None:
    target_ip = next(
        (ip for ip in env_sim.global_state.all_hosts if not ip.startswith('169.254.'))
    )
    blue_client = MockLLMClient(replies=[f'ACTION 0 TARGET {target_ip}'])
    res = run_episode(env_sim, {'blue_dmz': blue_client}, max_steps=3)
    assert isinstance(res, EpisodeResult)
    assert res.steps <= 3
    assert res.invalid_replies['blue_dmz'] == 0
    assert res.model_id == 'mock'


@pytest.mark.integration
def test_run_episode_random_client(env_sim) -> None:
    clients = {a: MockLLMClient(seed=i) for i, a in enumerate(env_sim.possible_agents)}
    res = run_episode(env_sim, clients, max_steps=4)
    assert res.steps <= 4
    assert sum(res.invalid_replies.values()) == 0


@pytest.mark.fast
def test_run_episode_counts_invalid_replies(env_sim) -> None:
    junk = MockLLMClient(replies=['I refuse.'])
    res = run_episode(env_sim, {'blue_dmz': junk}, max_steps=2)
    assert res.invalid_replies['blue_dmz'] == res.steps


@pytest.mark.fast
def test_leaderboard_roundtrip(tmp_path) -> None:
    path = tmp_path / 'leaderboard.json'
    for i in range(3):
        append_result(
            path,
            EpisodeResult(
                model_id='gpt-test',
                scenario='ransomware',
                steps=10,
                rewards={'blue_dmz': 5.0 + i, 'red_operator': -2.0},
                invalid_replies={'blue_dmz': 0},
                final_compromised=2,
                final_isolated=3,
            ),
        )
    summary = summarize(path)
    assert len(summary) == 1
    assert summary[0]['model_id'] == 'gpt-test'
    assert summary[0]['n_episodes'] == 3
    assert summary[0]['mean_total_reward'] == pytest.approx(
        sum((5.0 + i - 2.0 for i in range(3))) / 3
    )
