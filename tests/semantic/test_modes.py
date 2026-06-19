import pytest

pytest.importorskip('PIL')
import pytest
from netforge_rl.semantic.clients import MockLLMClient
from netforge_rl.semantic.modes import fm_vs_fm, zero_shot_attacker, zero_shot_defender


@pytest.mark.integration
def test_zero_shot_defender_runs(env_sim):
    blue = MockLLMClient(seed=0)
    res = zero_shot_defender(env_sim, blue, max_steps=4, seed=0)
    assert res['mode'] == 'zero_shot_defender'
    assert res['side'] == 'blue'
    assert res['model_id'] == 'mock'
    assert res['steps'] <= 4
    assert sum(res['invalid_replies'].values()) == 0


@pytest.mark.integration
def test_zero_shot_attacker_runs(env_sim):
    red = MockLLMClient(seed=1)
    res = zero_shot_attacker(env_sim, red, max_steps=4, seed=0)
    assert res['mode'] == 'zero_shot_attacker'
    assert res['side'] == 'red'
    assert 'red_operator' in res['cum_reward']


@pytest.mark.integration
def test_fm_vs_fm_records_both_models(env_sim):
    blue = MockLLMClient(seed=0)
    red = MockLLMClient(seed=1)
    res = fm_vs_fm(env_sim, blue, red, max_steps=4, seed=0)
    assert res['mode'] == 'fm_vs_fm'
    assert res['blue_model_id'] == 'mock'
    assert res['red_model_id'] == 'mock'
    assert 'red_operator' in res['cum_reward']
    assert 'blue_dmz' in res['cum_reward']


@pytest.mark.integration
def test_zero_shot_defender_counts_invalid_replies(env_sim):
    junk = MockLLMClient(replies=['I refuse.'])
    res = zero_shot_defender(env_sim, junk, max_steps=2, seed=0)
    n_blue = sum((1 for a in env_sim.possible_agents if a.startswith('blue')))
    assert sum(res['invalid_replies'].values()) >= res['steps'] * n_blue
