"""PyTree-registered JAX mirror of :class:`netforge_rl.core.functional.EnvState`.

The numpy-backed :class:`~netforge_rl.core.functional.HostArrays` is the
canonical authoring format; :class:`JaxEnvState` is the version that gets
batched under ``jax.vmap`` and traced by ``jax.jit``. Only the
vectorizable host fields and per-agent budget arrays become PyTree leaves;
string metadata travels as ``static_argnums`` / closure-captured aux data
(see :func:`to_jax`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from netforge_rl.core.functional import EnvState, HostArrays, HostMeta


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class JaxHostArrays:
    """JAX-leaf host SoA. Mirrors :class:`HostArrays` field-for-field."""

    status: jax.Array              # int8[N_HOSTS]
    privilege: jax.Array           # int8[N_HOSTS]
    decoy: jax.Array               # int8[N_HOSTS]
    edr_active: jax.Array          # bool[N_HOSTS]
    is_domain_controller: jax.Array
    contains_honeytokens: jax.Array
    human_vulnerability: jax.Array  # float32[N_HOSTS]
    cvss_score: jax.Array
    compromised_by_id: jax.Array   # int8[N_HOSTS], -1 == None
    system_integrity: jax.Array    # int8[N_HOSTS], code in INTEGRITY_CODES
    vuln_mask: jax.Array           # bool[N_HOSTS, N_CVE]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class JaxEnvState:
    """JAX PyTree mirror of :class:`EnvState`.

    Only the numeric fields are leaves. ``meta`` and ``agent_ids`` ride
    along as Python objects on a non-traced auxiliary slot — when this
    state goes through ``vmap`` they are not batched, by design.
    """

    hosts: JaxHostArrays
    agent_energy: jax.Array
    agent_funds: jax.Array
    agent_compute: jax.Array
    agent_locked_until: jax.Array
    current_tick: jax.Array              # int32 scalar
    business_downtime_score: jax.Array   # float32 scalar


# ── Conversion ─────────────────────────────────────────────────────────────


def to_jax(state: EnvState) -> JaxEnvState:
    """Move a numpy-backed :class:`EnvState` onto the default JAX device."""
    h = state.hosts
    jhosts = JaxHostArrays(
        status=jnp.asarray(h.status),
        privilege=jnp.asarray(h.privilege),
        decoy=jnp.asarray(h.decoy),
        edr_active=jnp.asarray(h.edr_active),
        is_domain_controller=jnp.asarray(h.is_domain_controller),
        contains_honeytokens=jnp.asarray(h.contains_honeytokens),
        human_vulnerability=jnp.asarray(h.human_vulnerability),
        cvss_score=jnp.asarray(h.cvss_score),
        compromised_by_id=jnp.asarray(h.compromised_by_id),
        system_integrity=jnp.asarray(h.system_integrity),
        vuln_mask=jnp.asarray(h.vuln_mask),
    )
    return JaxEnvState(
        hosts=jhosts,
        agent_energy=jnp.asarray(state.agent_energy),
        agent_funds=jnp.asarray(state.agent_funds),
        agent_compute=jnp.asarray(state.agent_compute),
        agent_locked_until=jnp.asarray(state.agent_locked_until),
        current_tick=jnp.asarray(state.current_tick, dtype=jnp.int32),
        business_downtime_score=jnp.asarray(
            state.business_downtime_score, dtype=jnp.float32
        ),
    )


def to_numpy(jstate: JaxEnvState, meta: HostMeta, agent_ids, knowledge=(), inventory=()) -> EnvState:
    """Materialize a numpy-backed :class:`EnvState` from a :class:`JaxEnvState`.

    String metadata (``meta``, ``agent_ids``, ``knowledge``, ``inventory``)
    must be supplied separately since it isn't carried by the JAX PyTree.
    """
    h = jstate.hosts
    hosts = HostArrays(
        status=np.asarray(h.status),
        privilege=np.asarray(h.privilege),
        decoy=np.asarray(h.decoy),
        edr_active=np.asarray(h.edr_active),
        is_domain_controller=np.asarray(h.is_domain_controller),
        contains_honeytokens=np.asarray(h.contains_honeytokens),
        human_vulnerability=np.asarray(h.human_vulnerability),
        cvss_score=np.asarray(h.cvss_score),
        compromised_by_id=np.asarray(h.compromised_by_id),
        system_integrity=np.asarray(h.system_integrity),
        vuln_mask=np.asarray(h.vuln_mask),
    )
    return EnvState(
        hosts=hosts,
        meta=meta,
        agent_ids=tuple(agent_ids),
        agent_energy=np.asarray(jstate.agent_energy),
        agent_funds=np.asarray(jstate.agent_funds),
        agent_compute=np.asarray(jstate.agent_compute),
        agent_locked_until=np.asarray(jstate.agent_locked_until),
        current_tick=int(jstate.current_tick),
        business_downtime_score=float(jstate.business_downtime_score),
        knowledge=tuple(knowledge),
        inventory=tuple(inventory),
    )
