from typing import NamedTuple

import jax
import jax.numpy as jnp

from netforge_rl.backends.jax.scenario_config import _BLUE_SCALE, _RED_SCALE, _RW
from netforge_rl.backends.jax.state_codes import _STATUS_ONLINE


class StepEvents(NamedTuple):
    """Per-tick reward-relevant events produced by the transition kernel."""

    # Red events
    red_writes_user: jax.Array
    red_writes_root: jax.Array
    new_red_impacts: jax.Array
    new_red_kinetics: jax.Array
    new_dc_compromise: jax.Array
    red_new_intel: jax.Array
    red_is_share: jax.Array
    red_writes_kill: jax.Array
    red_trapped: jax.Array
    looted_per_agent: jax.Array
    n_cve_compromises: jax.Array
    n_exfil_hosts: jax.Array
    # Blue events
    new_blue_isolations: jax.Array
    blue_writes_any_restore: jax.Array
    blue_writes_sat: jax.Array
    blue_is_rotate: jax.Array
    blue_new_intel: jax.Array
    new_decoy_deploys: jax.Array
    new_honey_deploys: jax.Array
    new_acl_writes: jax.Array
    new_misinform: jax.Array


def compute_rewards(state, ev: StepEvents, spec):
    """Score a tick's events into a per-agent reward vector [red..., blue...]."""
    rw_red, rw_blue = _RW[spec.scenario]
    w_user, w_root, w_impact, w_kinetic, w_exfil, w_dc, w_recon = rw_red
    w_good, w_bad, w_restore, w_health, w_dcloss, w_deceive = rw_blue

    target_clean = state.hosts.compromised_by_id < 0
    bad_isolations = ev.new_blue_isolations & target_clean
    good_isolations = ev.new_blue_isolations & ~target_clean

    n_hosts_f = jnp.float32(spec.n_hosts)
    healthy_ratio = (
        jnp.sum(
            (
                (state.hosts.compromised_by_id < 0)
                & (state.hosts.status == jnp.int8(_STATUS_ONLINE))
            ).astype(jnp.float32)
        )
        / n_hosts_f
    )
    dc_lost = jnp.any(
        state.hosts.is_domain_controller & (state.hosts.compromised_by_id >= 0)
    ).astype(jnp.float32)

    raw_blue = (
        w_good * jnp.sum(good_isolations.astype(jnp.float32))
        - w_bad * jnp.sum(bad_isolations.astype(jnp.float32))
        + w_restore * jnp.sum(ev.blue_writes_any_restore.astype(jnp.float32))
        + w_health * healthy_ratio
        - w_dcloss * dc_lost
        + w_deceive
        * jnp.sum(
            (
                ev.new_decoy_deploys
                | ev.new_honey_deploys
                | ev.new_acl_writes
                | ev.new_misinform
            ).astype(jnp.float32)
        )
        + 0.3 * jnp.sum(ev.blue_writes_sat.astype(jnp.float32))
        + 0.2 * jnp.sum(ev.blue_new_intel.astype(jnp.float32))
        + 4.0 * jnp.sum(ev.blue_is_rotate.astype(jnp.float32))
    )
    blue_reward = jnp.tanh(raw_blue / _BLUE_SCALE)

    raw_red = (
        w_user * jnp.sum(ev.red_writes_user.astype(jnp.float32))
        + 0.5 * ev.n_cve_compromises.astype(jnp.float32)
        + w_root * jnp.sum(ev.red_writes_root.astype(jnp.float32))
        + w_impact * jnp.sum(ev.new_red_impacts.astype(jnp.float32))
        + w_kinetic * jnp.sum(ev.new_red_kinetics.astype(jnp.float32))
        + w_dc * jnp.sum(ev.new_dc_compromise.astype(jnp.float32))
        + w_recon * jnp.sum(ev.red_new_intel.astype(jnp.float32))
        + w_exfil * ev.n_exfil_hosts
        + 0.4 * jnp.sum(ev.red_is_share.astype(jnp.float32))
        + 2.0 * jnp.sum(jnp.any(ev.looted_per_agent, axis=-1).astype(jnp.float32))
        + 5.0 * jnp.sum(ev.red_writes_kill.astype(jnp.float32))
    )
    red_team_reward = jnp.tanh(raw_red / _RED_SCALE)
    red_trap_penalty = -0.5 * ev.red_trapped.astype(jnp.float32)

    red_rewards = jnp.broadcast_to(red_team_reward, (spec.n_red,)) + red_trap_penalty
    blue_rewards = jnp.broadcast_to(blue_reward, (spec.n_blue,))
    return jnp.concatenate([red_rewards, blue_rewards])
