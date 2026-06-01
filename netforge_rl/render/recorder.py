"""Frame buffer + writer for evaluation reels."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FrameRecorder:
    """In-memory frame buffer. Writes mp4/gif via moviepy on flush.

    moviepy is imported lazily so the renderer dependency is optional.
    """

    def __init__(self, fps: int = 4):
        self.fps = fps
        self._frames: list[np.ndarray] = []

    def append(self, frame: np.ndarray) -> None:
        self._frames.append(frame)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frames(self) -> list[np.ndarray]:
        return self._frames

    def save(self, path: str | Path) -> Path:
        if not self._frames:
            raise ValueError('FrameRecorder has no frames to save.')
        from moviepy import ImageSequenceClip  # lazy

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        clip = ImageSequenceClip([f for f in self._frames], fps=self.fps)
        if path.suffix == '.gif':
            clip.write_gif(str(path), logger=None)
        else:
            clip.write_videofile(str(path), codec='libx264', logger=None)
        return path
