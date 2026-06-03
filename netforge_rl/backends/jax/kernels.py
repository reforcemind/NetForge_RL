from dataclasses import replace

import jax
import jax.numpy as jnp

from netforge_rl.backends.jax.state import JaxEnvState


def apply_host_status_delta(state: JaxEnvState, host_idx, status_code) -> JaxEnvState:
    new_status = state.hosts.status.at[host_idx].set(status_code)
    return replace(state, hosts=replace(state.hosts, status=new_status))


def apply_host_privilege_delta(state: JaxEnvState, host_idx, priv_code) -> JaxEnvState:
    new_priv = state.hosts.privilege.at[host_idx].set(priv_code)
    return replace(state, hosts=replace(state.hosts, privilege=new_priv))


def apply_compromised_by_delta(state: JaxEnvState, host_idx, agent_code) -> JaxEnvState:
    """Set compromised_by_id[host_idx] = agent_code; pass -1 to clear."""
    new_arr = state.hosts.compromised_by_id.at[host_idx].set(agent_code)
    return replace(state, hosts=replace(state.hosts, compromised_by_id=new_arr))


def resolve_conflicts_mask(red_target_mask, blue_target_mask, red_success, blue_success):
    """Return the post-resolution Red success vector — Red i is nullified iff
    it targets a host any successful Blue agent also targets.

    All inputs are leading-batchable so the function vmaps cleanly over envs.
    """
    defended_hosts = jnp.any(blue_target_mask & blue_success[:, None], axis=0)
    red_collisions = jnp.any(red_target_mask & defended_hosts[None, :], axis=1)
    return red_success & ~red_collisions
