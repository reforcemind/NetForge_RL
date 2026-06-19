import jax.numpy as jnp


def resolve_conflicts_mask(
    red_target_mask, blue_target_mask, red_success, blue_success
):
    """Return post-resolution Red success vector. Red success is nullified if targeted by Blue."""
    defended_hosts = jnp.any(blue_target_mask & blue_success[:, None], axis=0)
    red_collisions = jnp.any(red_target_mask & defended_hosts[None, :], axis=1)
    return red_success & ~red_collisions
