import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
torch = pytest.importorskip('torch')
import numpy as np
from netforge_rl.bridges.dlpack import jax_to_torch, torch_to_jax


@pytest.mark.fast
def test_jax_to_torch_round_trip_values() -> None:
    x_jax = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    t = jax_to_torch(x_jax)
    assert isinstance(t, torch.Tensor)
    np.testing.assert_array_equal(t.cpu().numpy(), np.asarray(x_jax))


@pytest.mark.fast
def test_torch_to_jax_round_trip_values() -> None:
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    x_jax = torch_to_jax(t)
    np.testing.assert_array_equal(np.asarray(x_jax), t.cpu().numpy())


@pytest.mark.fast
def test_torch_to_jax_handles_non_contiguous() -> None:
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    assert not t.is_contiguous()
    x_jax = torch_to_jax(t)
    np.testing.assert_array_equal(np.asarray(x_jax), t.contiguous().numpy())


@pytest.mark.fast
def test_torch_to_jax_handles_requires_grad() -> None:
    t = torch.arange(6, dtype=torch.float32, requires_grad=True)
    x_jax = torch_to_jax(t)
    assert int(x_jax[3]) == 3
