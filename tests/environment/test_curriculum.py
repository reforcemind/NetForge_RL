from netforge_rl.environment.curriculum import CurriculumWrapper, PhaseConfig


def test_curriculum_wrapper_init():
    phases = [
        PhaseConfig(
            name='test_phase_1',
            max_active_hosts=5,
            scenario_types=['ransomware'],
            reward_scale=1.0,
            dhcp_interval=0,
            topology_churn_rate=0.0,
            topology_migration_rate=0.0,
            topology_arrival_rate=0.0,
            advance_threshold=10.0,
            advance_window=2,
        ),
        PhaseConfig(
            name='test_phase_2',
            max_active_hosts=10,
            scenario_types=['apt_espionage'],
            reward_scale=2.0,
            dhcp_interval=10,
            topology_churn_rate=0.1,
            topology_migration_rate=0.1,
            topology_arrival_rate=0.1,
            advance_threshold=None,
            advance_window=2,
        ),
    ]

    wrapper = CurriculumWrapper(phases=phases)
    assert wrapper.phase_index == 0
    assert wrapper.phase.name == 'test_phase_1'


def test_curriculum_wrapper_properties():
    wrapper = CurriculumWrapper()
    assert 'red_operator' in wrapper.possible_agents
    assert 'red_operator' in wrapper.agents
    assert wrapper.observation_spaces
    assert wrapper.action_spaces
    assert wrapper.observation_space('red_operator')
    assert wrapper.action_space('red_operator')


def test_curriculum_wrapper_reset_and_step():
    phases = [
        PhaseConfig(
            name='test_phase_1',
            max_active_hosts=5,
            scenario_types=['ransomware'],
            reward_scale=2.0,
            dhcp_interval=0,
            topology_churn_rate=0.0,
            topology_migration_rate=0.0,
            topology_arrival_rate=0.0,
            advance_threshold=1.0,
            advance_window=1,
        ),
        PhaseConfig(
            name='test_phase_2',
            max_active_hosts=10,
            scenario_types=['apt_espionage'],
            reward_scale=1.0,
            dhcp_interval=10,
            topology_churn_rate=0.1,
            topology_migration_rate=0.1,
            topology_arrival_rate=0.1,
            advance_threshold=None,
            advance_window=1,
        ),
    ]

    advanced_called = False

    def on_advance(idx, name):
        nonlocal advanced_called
        advanced_called = True

    wrapper = CurriculumWrapper(phases=phases, on_phase_advance=on_advance)
    obs, info = wrapper.reset()
    assert 'red_operator' in obs
    assert '__curriculum__' in info['red_operator']

    actions = {a: wrapper.action_space(a).sample() for a in wrapper.agents}
    obs, rewards, term, trunc, infos = wrapper.step(actions)

    assert '__curriculum__' in infos['red_operator']
    assert not infos['red_operator']['__curriculum__']['phase_advanced']

    while not (all(term.values()) or all(trunc.values())):
        actions = {a: wrapper.action_space(a).sample() for a in wrapper.agents}
        wrapper._episode_reward = 100.0
        obs, rewards, term, trunc, infos = wrapper.step(actions)

    assert wrapper.phase_index == 1
    assert wrapper.phase.name == 'test_phase_2'
    assert advanced_called


def test_curriculum_wrapper_render_and_close():
    import pytest

    pytest.importorskip('matplotlib')
    wrapper = CurriculumWrapper()
    img = wrapper.render()
    assert img is not None
    wrapper.close()
