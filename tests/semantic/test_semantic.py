import base64
import numpy as np
import pytest
from netforge_rl.core.functional import from_global_state
from netforge_rl.semantic import (
    action_menu,
    build_vla_prompt,
    parse_action,
    state_to_text,
)

AGENTS = ('red_operator', 'blue_dmz', 'blue_internal', 'blue_restricted')


@pytest.mark.fast
def test_action_menu_contains_team_actions() -> None:
    blue_menu = action_menu('blue_dmz')
    red_menu = action_menu('red_operator')
    assert 'IsolateHost' in blue_menu.values()
    assert 'ExploitRemoteService' in red_menu.values()
    assert 'ExfiltrateData' in red_menu.values()


@pytest.mark.fast
def test_state_to_text_renders_for_both_roles(global_state) -> None:
    snap = from_global_state(global_state, AGENTS)
    blue = state_to_text(snap, 'blue_dmz', max_hosts=4)
    red = state_to_text(snap, 'red_operator', max_hosts=4)
    assert 'Blue SOC Operator' in blue
    assert 'Red Operator' in red
    assert 'NetForge SIEM Report' in blue
    assert 'Legal actions' in blue
    assert 'ACTION' in blue and 'TARGET' in blue


@pytest.mark.fast
def test_state_to_text_filters_padding_hosts(global_state) -> None:
    snap = from_global_state(global_state, AGENTS)
    report = state_to_text(snap, 'blue_dmz', max_hosts=200, include_menu=False)
    assert '169.254.' not in report


@pytest.mark.fast
def test_state_to_text_includes_budgets(env_sim) -> None:
    snap = env_sim.to_envstate()
    report = state_to_text(snap, 'blue_dmz')
    assert 'energy=' in report and 'funds=' in report


@pytest.mark.fast
def test_parser_happy_path() -> None:
    ips = ['10.0.0.1', '10.0.0.2', '192.168.1.5']
    out = parse_action('Plan: ACTION 0 TARGET 10.0.0.2 # isolate', 'blue_dmz', ips)
    assert out == (0, 1)


@pytest.mark.fast
def test_parser_rejects_unknown_action() -> None:
    ips = ['10.0.0.1']
    assert parse_action('ACTION 99 TARGET 10.0.0.1', 'blue_dmz', ips) is None


@pytest.mark.fast
def test_parser_rejects_unknown_target() -> None:
    ips = ['10.0.0.1']
    assert parse_action('ACTION 0 TARGET 8.8.8.8', 'blue_dmz', ips) is None


@pytest.mark.fast
def test_parser_returns_none_on_garbage() -> None:
    assert parse_action('I will think about it.', 'blue_dmz', ['10.0.0.1']) is None


@pytest.mark.fast
def test_build_vla_prompt_round_trip() -> None:
    rgb = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    prompt = build_vla_prompt(rgb, 'hello')
    assert prompt['text'] == 'hello'
    assert prompt['mime_type'] == 'image/png'
    raw = base64.b64decode(prompt['image_b64_png'])
    assert raw.startswith(b'\x89PNG')


@pytest.mark.fast
def test_build_vla_prompt_validates_input() -> None:
    with pytest.raises(ValueError):
        build_vla_prompt(np.zeros((8, 8), dtype=np.uint8), 'x')
    with pytest.raises(ValueError):
        build_vla_prompt(np.zeros((8, 8, 3), dtype=np.float32), 'x')


@pytest.mark.integration
def test_end_to_end_text_action_loop(env_sim) -> None:
    snap = env_sim.to_envstate()
    report = state_to_text(snap, 'blue_dmz', max_hosts=4)
    target_ips = sorted(env_sim.global_state.all_hosts.keys())
    real_ip = next((ip for ip in target_ips if not ip.startswith('169.254.')))
    fake_reply = f'After reading:\n{report[:50]}\nACTION 0 TARGET {real_ip}'
    parsed = parse_action(fake_reply, 'blue_dmz', target_ips)
    assert parsed is not None
    action_type, target_idx = parsed
    actions = {agent: np.array([0, 0], dtype=np.int64) for agent in env_sim.agents}
    actions['blue_dmz'] = np.array([action_type, target_idx], dtype=np.int64)
    env_sim.step(actions)
