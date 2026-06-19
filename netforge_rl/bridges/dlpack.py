import jax
import jax.dlpack
import torch
import torch.utils.dlpack


def jax_to_torch(arr):
    """Zero-copy jax.Array -> torch.Tensor via DLPack."""
    return torch.utils.dlpack.from_dlpack(arr)


def torch_to_jax(t):
    """Zero-copy torch.Tensor -> jax.Array; detaches and contiguousifies first."""
    if t.requires_grad:
        t = t.detach()
    if not t.is_contiguous():
        t = t.contiguous()
    return jax.dlpack.from_dlpack(t)
