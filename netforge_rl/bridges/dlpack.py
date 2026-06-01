"""Zero-copy ``jax.Array`` <-> ``torch.Tensor`` via DLPack.

Lets CleanRL / Stable-Baselines3 / RLlib consume JAX-vectorized rollouts
without leaving the device. ``torch`` is required; the module's imports
fail loudly if it isn't installed.
"""

from __future__ import annotations

from importlib.util import find_spec

if find_spec('torch') is None:  # pragma: no cover
    raise ImportError(
        'netforge_rl.bridges.dlpack requires torch. '
        "Install with `pip install torch`."
    )

import jax
import jax.dlpack
import torch
import torch.utils.dlpack


def jax_to_torch(arr: jax.Array) -> torch.Tensor:
    """Wrap a JAX array as a torch Tensor with shared storage when possible."""
    return torch.utils.dlpack.from_dlpack(arr)


def torch_to_jax(t: torch.Tensor) -> jax.Array:
    """Wrap a torch Tensor as a JAX array with shared storage when possible.

    Non-contiguous / requires_grad tensors are contiguous'd / detached
    first (DLPack does not support those).
    """
    if t.requires_grad:
        t = t.detach()
    if not t.is_contiguous():
        t = t.contiguous()
    return jax.dlpack.from_dlpack(t)
