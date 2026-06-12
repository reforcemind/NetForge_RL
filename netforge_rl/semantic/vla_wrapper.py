import base64
from io import BytesIO
from PIL import Image

import numpy as np


def _png_b64(rgb):
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def build_vla_prompt(rgb, text):
    """Return a provider-neutral ``{'text', 'image_b64_png', 'mime_type'}`` payload."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('expected uint8 HxWx3 RGB array')
    return {
        'text': text,
        'image_b64_png': _png_b64(rgb),
        'mime_type': 'image/png',
    }
