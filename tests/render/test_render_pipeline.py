"""Render pipeline smoke tests — snapshot, RGB array shape, recorder, env hook."""

import numpy as np
import pytest

from netforge_rl.render import FrameRecorder, render_rgb, snapshot_from_envstate
from netforge_rl.render.snapshot import (
    COLOR_COMPROMISED,
    COLOR_ISOLATED,
    COLOR_SECURE,
)


AGENTS = ('red_operator', 'blue_dmz', 'blue_internal', 'blue_restricted')


@pytest.mark.fast
def test_snapshot_filters_padding_hosts(global_state) -> None:
    from netforge_rl.core.functional import from_global_state

    snap = snapshot_from_envstate(from_global_state(global_state, AGENTS))
    assert snap.n_nodes < 100
    assert all(not s.startswith('169.254.') for s in snap.subnets)


@pytest.mark.fast
def test_snapshot_color_classifications() -> None:
    """Status / privilege / honeytoken should map to the documented palette."""
    from dataclasses import replace
    import numpy as np
    from netforge_rl.core.functional import (
        from_global_state,
        STATUS_CODES,
        PRIVILEGE_CODES,
    )

    # Build a synthetic 3-host state and check the classifier directly.
    from netforge_rl.core.state import GlobalNetworkState, Host, Subnet

    gs = GlobalNetworkState()
    gs.add_subnet(Subnet('10.0.0.0/24', 'X'))
    for ip in ('10.0.0.1', '10.0.0.2', '10.0.0.3'):
        gs.register_host(Host(ip, ip, '10.0.0.0/24'))
    # Pad to 100.
    for p in range(97):
        gs.register_host(Host(f'169.254.0.{p}', f'pad{p}', '169.254.0.0/16'))
    gs.all_hosts['10.0.0.2'].privilege = 'Root'           # compromised
    gs.all_hosts['10.0.0.3'].status = 'isolated'

    snap = snapshot_from_envstate(from_global_state(gs, AGENTS))
    # Active subnets are X (3 hosts) only.
    assert snap.n_nodes == 3
    np.testing.assert_allclose(snap.colors[0], COLOR_SECURE, atol=1e-3)
    np.testing.assert_allclose(snap.colors[1], COLOR_COMPROMISED, atol=1e-3)
    np.testing.assert_allclose(snap.colors[2], COLOR_ISOLATED, atol=1e-3)


@pytest.mark.fast
def test_render_rgb_shape(global_state) -> None:
    from netforge_rl.core.functional import from_global_state

    snap = snapshot_from_envstate(from_global_state(global_state, AGENTS))
    img = render_rgb(snap, figsize=(3.0, 3.0), dpi=80)
    assert img.shape == (240, 240, 3)
    assert img.dtype == np.uint8


@pytest.mark.fast
def test_frame_recorder_append_and_len() -> None:
    rec = FrameRecorder(fps=4)
    rec.append(np.zeros((10, 10, 3), dtype=np.uint8))
    rec.append(np.ones((10, 10, 3), dtype=np.uint8))
    assert len(rec) == 2


@pytest.mark.fast
def test_frame_recorder_save_requires_frames(tmp_path) -> None:
    rec = FrameRecorder()
    with pytest.raises(ValueError):
        rec.save(tmp_path / 'empty.gif')


@pytest.mark.integration
def test_env_render_rgb_returns_image(env_sim) -> None:
    img = env_sim.render(mode='rgb_array')
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3 and img.shape[2] == 3
