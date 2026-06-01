"""Vision-Language-Action wrapper: bundles (image, text) for multimodal models.

Returns a provider-neutral dict that downstream client adapters convert
into their native message format (Anthropic ``content`` blocks, OpenAI
``messages[*].content`` lists, Gemini ``parts``, etc.).
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np


def _png_b64(rgb: np.ndarray) -> str:
    from PIL import Image  # lazy — Pillow ships with matplotlib

    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def build_vla_prompt(rgb: np.ndarray, text: str) -> dict[str, Any]:
    """Return a ``{'text': ..., 'image_b64_png': ...}`` payload."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('expected uint8 HxWx3 RGB array')
    return {
        'text': text,
        'image_b64_png': _png_b64(rgb),
        'mime_type': 'image/png',
    }
