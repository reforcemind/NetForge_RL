from netforge_rl.render.replay_html import ReplayLog, write_replay_html
from netforge_rl.render.snapshot import Snapshot, snapshot_from_envstate
from netforge_rl.render.trajectory import TrajectoryRecorder

__all__ = [
    'FrameRecorder',
    'ReplayLog',
    'Snapshot',
    'TrajectoryRecorder',
    'render_rgb',
    'snapshot_from_envstate',
    'write_replay_html',
]


def __getattr__(name):
    if name == 'FrameRecorder':
        from netforge_rl.render.recorder import FrameRecorder

        return FrameRecorder
    if name == 'render_rgb':
        from netforge_rl.render.matplotlib_renderer import render_rgb

        return render_rgb
    raise AttributeError(name)
