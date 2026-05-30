"""Frozen, PyTree-compatible environment state — Phase 1 scaffolding.

This module introduces the immutable struct-of-arrays representation that
will become the canonical state object once the JAX backend lands in Phase 2.
For now it coexists with the legacy mutable :class:`GlobalNetworkState`, and a
round-trip-tested converter (:func:`from_global_state` / :func:`to_global_state`)
lets us shift consumers over incrementally without breaking the legacy
PettingZoo path.

Design choices:

* **Numeric leaves only.** All vectorizable fields live as ``np.ndarray`` of
  length ``N_HOSTS`` so they can later be marked as JAX PyTree leaves and
  ``jax.vmap``'d over a batch axis without restructuring.
* **Categorical encoding.** ``status``, ``privilege``, ``decoy`` are stored as
  small integer codes against the codebooks below. The codebooks themselves
  are module-level constants so the integer↔string mapping is process-wide
  stable.
* **Static metadata is segregated.** Variable-length / string fields (``ip``,
  ``hostname``, ``services``, ``vulnerabilities``) live in
  :class:`HostMeta` as Python tuples/lists. These are *not* PyTree leaves;
  when the JAX backend arrives they will be carried as static auxiliary data
  on the PyTree so XLA never sees them.
* **No mutation.** Every dataclass is ``frozen=True``. State transitions in
  Phase 1.5+ will return a *new* :class:`EnvState` via :func:`dataclasses.replace`.

The converter is intentionally lossy in one direction only: round-tripping
through :class:`EnvState` preserves all fields currently consumed by the
environment loop, but discards a handful of legacy-only fields (action
history, pending effects, SIEM buffer) that the functional core will model
explicitly in later slices. The converter test pins this contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from netforge_rl.core.state import GlobalNetworkState


N_HOSTS = 100  # Fixed per existing NetworkGenerator contract; see state.py:212.


# ── Categorical codebooks ──────────────────────────────────────────────────
# Order is load-bearing — do not reorder without bumping a versioned hash.

STATUS_CODES = ('online', 'isolated')
PRIVILEGE_CODES = ('None', 'User', 'Root')
DECOY_CODES = ('inactive', 'active', 'Apache', 'SSHD', 'Tomcat')


def _encode(value: str, codebook: tuple[str, ...]) -> int:
    try:
        return codebook.index(value)
    except ValueError:
        return 0  # unknown -> first entry (None/inactive/online)


def _decode(code: int, codebook: tuple[str, ...]) -> str:
    return codebook[int(code)] if 0 <= int(code) < len(codebook) else codebook[0]


# ── Frozen state containers ────────────────────────────────────────────────


@dataclass(frozen=True)
class HostArrays:
    """Vectorizable per-host state — destined to become JAX PyTree leaves."""

    status: np.ndarray              # int8[N_HOSTS], code in STATUS_CODES
    privilege: np.ndarray           # int8[N_HOSTS], code in PRIVILEGE_CODES
    decoy: np.ndarray               # int8[N_HOSTS], code in DECOY_CODES
    edr_active: np.ndarray          # bool[N_HOSTS]
    is_domain_controller: np.ndarray  # bool[N_HOSTS]
    contains_honeytokens: np.ndarray  # bool[N_HOSTS]
    human_vulnerability: np.ndarray  # float32[N_HOSTS]
    cvss_score: np.ndarray          # float32[N_HOSTS]
    # compromised_by is stored as an integer ID against EnvState.agent_ids
    # (-1 == 'None' / not compromised). Strings are intentionally kept out of
    # the vectorizable leaves so XLA can fuse comparisons.
    compromised_by_id: np.ndarray   # int8[N_HOSTS]


@dataclass(frozen=True)
class HostMeta:
    """Static / variable-length host metadata. NOT a vectorizable leaf."""

    ip: tuple[str, ...]
    hostname: tuple[str, ...]
    subnet_cidr: tuple[str, ...]
    os: tuple[str, ...]
    services: tuple[tuple[str, ...], ...]
    vulnerabilities: tuple[tuple[str, ...], ...]
    cached_credentials: tuple[tuple[str, ...], ...]
    system_tokens: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class EnvState:
    """Single immutable snapshot of the MARL environment.

    Phase 1 scaffolding: produced by the converter, consumed by no-one yet.
    Phase 1.5+ will route the legacy ParallelEnv through this type.
    """

    hosts: HostArrays
    meta: HostMeta

    agent_ids: tuple[str, ...]              # canonical agent order
    agent_energy: np.ndarray                # int32[len(agent_ids)]
    agent_funds: np.ndarray                 # int32[len(agent_ids)]
    agent_compute: np.ndarray               # int32[len(agent_ids)]
    agent_locked_until: np.ndarray          # int32[len(agent_ids)]

    current_tick: int = 0
    business_downtime_score: float = 0.0

    # Per-agent fog-of-war: tuple of frozensets aligned to agent_ids.
    knowledge: tuple[frozenset[str], ...] = field(default_factory=tuple)
    # Per-agent stolen tokens / credentials, aligned to agent_ids.
    inventory: tuple[frozenset[str], ...] = field(default_factory=tuple)

    @property
    def host_count(self) -> int:
        return self.hosts.status.shape[0]

    def agent_index(self, agent_id: str) -> int:
        return self.agent_ids.index(agent_id)

    def host_index(self, ip: str) -> int:
        return self.meta.ip.index(ip)

    def with_tick(self, tick: int) -> 'EnvState':
        return replace(self, current_tick=tick)


# ── Converters ─────────────────────────────────────────────────────────────


def from_global_state(
    legacy: 'GlobalNetworkState',
    agent_ids: tuple[str, ...],
) -> EnvState:
    """Materialize a frozen ``EnvState`` from the legacy mutable state.

    Hosts are ordered by sorted IP — the same canonical order already used
    everywhere in the legacy env (e.g. ``parallel_env.py:222`` and
    ``state.get_adjacency_matrix``). This guarantees ``host_index(ip)`` agrees
    with the legacy ``sorted(state.all_hosts)`` ordering relied on by the
    action registry.
    """
    sorted_ips = tuple(sorted(legacy.all_hosts.keys()))
    n = len(sorted_ips)
    if n != N_HOSTS:
        raise ValueError(
            f'Expected exactly {N_HOSTS} hosts (legacy pads to this); got {n}.'
        )

    hosts_in_order = [legacy.all_hosts[ip] for ip in sorted_ips]

    status_arr = np.array(
        [_encode(h.status, STATUS_CODES) for h in hosts_in_order], dtype=np.int8
    )
    priv_arr = np.array(
        [_encode(h.privilege, PRIVILEGE_CODES) for h in hosts_in_order], dtype=np.int8
    )
    decoy_arr = np.array(
        [_encode(h.decoy, DECOY_CODES) for h in hosts_in_order], dtype=np.int8
    )
    edr_arr = np.array([bool(h.edr_active) for h in hosts_in_order], dtype=bool)
    dc_arr = np.array(
        [bool(h.is_domain_controller) for h in hosts_in_order], dtype=bool
    )
    honey_arr = np.array(
        [bool(getattr(h, 'contains_honeytokens', False)) for h in hosts_in_order],
        dtype=bool,
    )
    hvuln_arr = np.array(
        [float(h.human_vulnerability_score) for h in hosts_in_order], dtype=np.float32
    )
    cvss_arr = np.array(
        [float(getattr(h, 'cvss_score', 0.0)) for h in hosts_in_order],
        dtype=np.float32,
    )
    comp_arr = np.array(
        [
            agent_ids.index(h.compromised_by) if h.compromised_by in agent_ids else -1
            for h in hosts_in_order
        ],
        dtype=np.int8,
    )

    hosts = HostArrays(
        status=status_arr,
        privilege=priv_arr,
        decoy=decoy_arr,
        edr_active=edr_arr,
        is_domain_controller=dc_arr,
        contains_honeytokens=honey_arr,
        human_vulnerability=hvuln_arr,
        cvss_score=cvss_arr,
        compromised_by_id=comp_arr,
    )

    meta = HostMeta(
        ip=sorted_ips,
        hostname=tuple(h.hostname for h in hosts_in_order),
        subnet_cidr=tuple(h.subnet_cidr for h in hosts_in_order),
        os=tuple(h.os for h in hosts_in_order),
        services=tuple(tuple(h.services) for h in hosts_in_order),
        vulnerabilities=tuple(tuple(h.vulnerabilities) for h in hosts_in_order),
        cached_credentials=tuple(
            tuple(h.cached_credentials) for h in hosts_in_order
        ),
        system_tokens=tuple(tuple(h.system_tokens) for h in hosts_in_order),
    )

    energy = np.array(
        [int(legacy.agent_energy.get(a, 0)) for a in agent_ids], dtype=np.int32
    )
    funds = np.array(
        [int(legacy.agent_funds.get(a, 0)) for a in agent_ids], dtype=np.int32
    )
    compute = np.array(
        [int(legacy.agent_compute.get(a, 0)) for a in agent_ids], dtype=np.int32
    )
    locked = np.array(
        [int(legacy.agent_locked_until.get(a, 0)) for a in agent_ids], dtype=np.int32
    )

    knowledge = tuple(
        frozenset(legacy.agent_knowledge.get(a, set())) for a in agent_ids
    )
    inventory = tuple(
        frozenset(legacy.agent_inventory.get(a, set())) for a in agent_ids
    )

    return EnvState(
        hosts=hosts,
        meta=meta,
        agent_ids=tuple(agent_ids),
        agent_energy=energy,
        agent_funds=funds,
        agent_compute=compute,
        agent_locked_until=locked,
        current_tick=int(legacy.current_tick),
        business_downtime_score=float(legacy.business_downtime_score),
        knowledge=knowledge,
        inventory=inventory,
    )


def to_global_state(snap: EnvState) -> 'GlobalNetworkState':
    """Inverse of :func:`from_global_state`.

    Reconstructs a mutable :class:`GlobalNetworkState` from a frozen snapshot.
    Used during the migration window so functional-core code paths can hand
    off to legacy consumers that haven't been ported yet.
    """
    from netforge_rl.core.state import GlobalNetworkState, Subnet, Host

    legacy = GlobalNetworkState()

    # Recreate subnets in first-seen order; legacy code keys them by CIDR.
    seen: dict[str, Subnet] = {}
    for cidr in snap.meta.subnet_cidr:
        if cidr in seen:
            continue
        # Subnet name is not roundtrippable from the snapshot (we don't store
        # it). The legacy state only uses Subnet.name for diagnostics, never
        # for routing decisions — derive a stable label from the CIDR.
        sn = Subnet(cidr=cidr, name=cidr)
        seen[cidr] = sn
        legacy.add_subnet(sn)

    for i, ip in enumerate(snap.meta.ip):
        host = Host(
            ip=ip,
            hostname=snap.meta.hostname[i],
            subnet_cidr=snap.meta.subnet_cidr[i],
        )
        host.status = _decode(snap.hosts.status[i], STATUS_CODES)
        host.privilege = _decode(snap.hosts.privilege[i], PRIVILEGE_CODES)
        host.decoy = _decode(snap.hosts.decoy[i], DECOY_CODES)
        host.edr_active = bool(snap.hosts.edr_active[i])
        host.is_domain_controller = bool(snap.hosts.is_domain_controller[i])
        host.contains_honeytokens = bool(snap.hosts.contains_honeytokens[i])
        host.human_vulnerability_score = float(snap.hosts.human_vulnerability[i])
        host.cvss_score = float(snap.hosts.cvss_score[i])
        host.os = snap.meta.os[i]
        host.services = list(snap.meta.services[i])
        host.vulnerabilities = list(snap.meta.vulnerabilities[i])
        host.cached_credentials = list(snap.meta.cached_credentials[i])
        host.system_tokens = list(snap.meta.system_tokens[i])
        cid = int(snap.hosts.compromised_by_id[i])
        host.compromised_by = snap.agent_ids[cid] if cid >= 0 else 'None'
        legacy.register_host(host)

    for j, agent in enumerate(snap.agent_ids):
        legacy.agent_energy[agent] = int(snap.agent_energy[j])
        legacy.agent_funds[agent] = int(snap.agent_funds[j])
        legacy.agent_compute[agent] = int(snap.agent_compute[j])
        legacy.agent_locked_until[agent] = int(snap.agent_locked_until[j])
        legacy.agent_knowledge[agent] = set(snap.knowledge[j])
        legacy.agent_inventory[agent] = set(snap.inventory[j])

    legacy.current_tick = int(snap.current_tick)
    legacy.business_downtime_score = float(snap.business_downtime_score)

    return legacy
