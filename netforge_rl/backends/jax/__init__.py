"""JAX backend — vectorized, jit-friendly kernels for ``EnvState``.

Importing this package fails with a clear message if JAX isn't installed.
Install the optional dependency with ``pip install netforge_rl[jax]``.
"""

from importlib.util import find_spec


if find_spec('jax') is None:  # pragma: no cover - environment guard
    raise ImportError(
        'netforge_rl.backends.jax requires JAX. Install with '
        "`pip install 'netforge_rl[jax]'` or `pip install 'jax[cpu]'`."
    )


from netforge_rl.backends.jax.state import JaxEnvState, to_jax, to_numpy  # noqa: E402
from netforge_rl.backends.jax.kernels import (  # noqa: E402
    apply_host_status_delta,
    apply_host_privilege_delta,
    apply_compromised_by_delta,
    resolve_conflicts_mask,
)
from netforge_rl.backends.jax.vector_env import (  # noqa: E402
    BatchedActions,
    VectorEnvSpec,
    initial_batched_state,
    make_vector_step,
    random_actions,
)

__all__ = [
    'BatchedActions',
    'JaxEnvState',
    'VectorEnvSpec',
    'apply_compromised_by_delta',
    'apply_host_privilege_delta',
    'apply_host_status_delta',
    'initial_batched_state',
    'make_vector_step',
    'random_actions',
    'resolve_conflicts_mask',
    'to_jax',
    'to_numpy',
]
