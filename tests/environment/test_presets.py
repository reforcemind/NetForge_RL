import pytest

from netforge_rl.environment import (
    DIFFICULTY_PRESETS,
    EVAL_SEEDS,
    make_config,
    make_env,
)
from netforge_rl.environment.parallel_env import PADDING_SUBNET


def _active(env):
    return [
        h
        for h in env.global_state.all_hosts.values()
        if h.subnet_cidr != PADDING_SUBNET
    ]


@pytest.mark.fast
def test_presets_expose_three_tiers():
    assert set(DIFFICULTY_PRESETS) == {'easy', 'medium', 'hard'}


@pytest.mark.fast
def test_difficulty_ladder_is_monotonic():
    sizes = {d: len(_active(make_env(d, seed=0))) for d in ('easy', 'medium', 'hard')}
    assert sizes['easy'] < sizes['hard']
    assert sizes['easy'] <= sizes['medium'] <= sizes['hard']


@pytest.mark.fast
def test_knobs_are_applied():
    env = make_env('hard', seed=0)
    assert env.log_latency == 4
    assert env.dhcp_interval == 40
    easy = make_env('easy', seed=0)
    assert easy.log_latency == 0
    assert easy.dhcp_interval == 0


@pytest.mark.fast
def test_overrides_win_over_preset():
    cfg = make_config(
        'easy', scenario_type='apt_espionage', max_ticks=17, log_latency=9
    )
    assert cfg['scenario_type'] == 'apt_espionage'
    assert cfg['max_ticks'] == 17
    assert cfg['log_latency'] == 9


@pytest.mark.fast
def test_unknown_difficulty_raises():
    with pytest.raises(KeyError):
        make_config('impossible')


@pytest.mark.fast
def test_evaluation_mode_uses_held_out_topology():
    """Held-out seeds must produce a different topology than the same train seed."""
    train = make_env('medium', evaluation=False, seed=EVAL_SEEDS[0])
    held = make_env('medium', evaluation=True, seed=EVAL_SEEDS[0])
    assert set(train.global_state.all_hosts) != set(held.global_state.all_hosts)


@pytest.mark.fast
def test_eval_seed_suite_is_frozen():
    assert len(EVAL_SEEDS) == 20
    assert len(set(EVAL_SEEDS)) == 20  # no duplicates
