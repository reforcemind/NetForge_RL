from pathlib import Path

import numpy as np
try:
    from moviepy import ImageSequenceClip
except ImportError:
    pass


class FrameRecorder:
    """In-memory frame buffer; writes mp4/gif via moviepy on save()."""

    def __init__(self, fps=4):
        self.fps = fps
        self._frames = []

    def append(self, frame):
        self._frames.append(frame)

    def __len__(self):
        return len(self._frames)

    @property
    def frames(self):
        return self._frames

    def save(self, path):
        if not self._frames:
            raise ValueError('FrameRecorder has no frames to save.')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        clip = ImageSequenceClip([f for f in self._frames], fps=self.fps)
        if path.suffix == '.gif':
            clip.write_gif(str(path), logger=None)
        else:
            clip.write_videofile(str(path), codec='libx264', logger=None)
        return path
