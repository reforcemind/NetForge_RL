import pytest

from netforge_rl.baselines.policies import HeuristicBluePolicy, RandomPolicy
from netforge_rl.comm.wrapper import CommFailureWrapper
from netforge_rl.datasets.collector import collect_episode
from netforge_rl.datasets.minari_export import export_npz, trajectories_to_minari_dict
from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.league.psro import payoff_matrix
from netforge_rl.render.replay_html import ReplayLog, write_replay_html
from netforge_rl.rewards.variants import RewardDesignWrapper


@pytest.mark.fast
def test_comm_drop_zeros_channel():
    env = CommFailureWrapper(
        NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 4}),
        drop_p=1.0,
        seed=0,
    )
    obs, _ = env.reset(seed=0)
    obs, *_ = env.step({a: env.action_space(a).sample() for a in env.agents})
    assert float(obs['blue_dmz']['blue_comm'].sum()) == 0.0


@pytest.mark.fast
def test_reward_variant_records_raw():
    env = RewardDesignWrapper(
        NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 4}),
        variant='sla_only',
    )
    env.reset(seed=0)
    _, rewards, _, _, infos = env.step(
        {a: env.action_space(a).sample() for a in env.agents}
    )
    assert 'raw_reward' in infos['blue_dmz']
    assert infos['blue_dmz']['reward_variant'] == 'sla_only'
    assert 'blue_dmz' in rewards


@pytest.mark.integration
def test_offline_export_npz(tmp_path):
    traj = collect_episode(
        RandomPolicy(seed=0),
        HeuristicBluePolicy(seed=0),
        max_ticks=8,
        seed=0,
        quality='heuristic',
    )
    payload = trajectories_to_minari_dict([traj])
    assert payload['n_episodes'] == 1
    path = export_npz([traj], tmp_path / 'ds.npz')
    assert path.exists()


@pytest.mark.fast
def test_try_minari_export_writes_npz_or_hdf5(tmp_path):
    from netforge_rl.datasets.minari_export import try_minari_export

    traj = collect_episode(
        RandomPolicy(seed=0),
        HeuristicBluePolicy(seed=0),
        max_ticks=4,
        seed=0,
        quality='heuristic',
    )
    path = try_minari_export([traj], 'netforge/test-v0', tmp_path / 'ds.hdf5')
    assert path.exists()
    assert path.suffix in {'.hdf5', '.npz'}


@pytest.mark.fast
def test_psro_matrix_shape():
    game = payoff_matrix(
        {'r': lambda: RandomPolicy(seed=0)},
        {'b': lambda: HeuristicBluePolicy(seed=0)},
        seeds=(0,),
        max_ticks=5,
    )
    assert len(game['payoff_blue']) == 1
    assert len(game['payoff_blue'][0]) == 1


@pytest.mark.fast
def test_replay_html(tmp_path):
    env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 4})
    env.reset(seed=0)
    log = ReplayLog('ransomware', 0)
    _, rewards, _, _, infos = env.step(
        {a: env.action_space(a).sample() for a in env.agents}
    )
    log.capture(env, rewards, infos, {})
    html = write_replay_html(log.to_dict(), tmp_path / 'replay.html')
    text = html.read_text(encoding='utf-8')
    assert 'NETFORGE' in text
    log.save_json(tmp_path / 'traj.json')
    assert (tmp_path / 'traj.json').exists()
