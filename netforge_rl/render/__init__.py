"""Decoupled rendering pipeline.

Render code never touches the env hot path. Consumers convert a frozen
:class:`~netforge_rl.core.functional.EnvState` to a CPU :class:`Snapshot`,
then pass the snapshot to a renderer.
"""

from netforge_rl.render.snapshot import Snapshot, snapshot_from_envstate
from netforge_rl.render.matplotlib_renderer import render_rgb
from netforge_rl.render.recorder import FrameRecorder

__all__ = [
    'FrameRecorder',
    'Snapshot',
    'render_rgb',
    'snapshot_from_envstate',
]
