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

STATUS_CODES = ('online', 'isolated', 'kernel_panic')
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


# ── Pure delta interpreter ─────────────────────────────────────────────────
#
# Each action in netforge_rl/actions/ emits an ``ActionEffect.state_deltas``
# entry, today consumed by the mutating ``GlobalNetworkState.apply_delta``.
# The functional core re-interprets those same string-keyed deltas against
# an immutable EnvState, returning a new EnvState. Keeping the wire format
# identical means the action layer needs zero changes — the imperative
# shell (Phase 1 slice 4) can pick whichever interpreter it likes.
#
# Supported keys (mirrors the cases in GlobalNetworkState.apply_delta):
#   ``hosts/<ip>/<attr>``        where <attr> is one of: status, privilege,
#                                decoy, edr_active, compromised_by,
#                                is_domain_controller, contains_honeytokens,
#                                human_vulnerability_score, cvss_score,
#                                os, services, vulnerabilities,
#                                cached_credentials, system_tokens
#   ``knowledge/<agent>/<ip>``   add <ip> to agent's fog-of-war set
#   ``history/<agent>/<record>`` (no-op here; action_history isn't in EnvState yet)
#   ``firewall/...``             (deferred — firewall isn't in EnvState yet)
#
# Unknown keys are silently ignored (matching legacy behavior). Command-
# object deltas (``hasattr(delta, 'execute')``) are out of scope; those will
# be migrated when their target state moves into EnvState.


# Attribute → (HostArrays field, encoder) for vectorizable per-host fields.
_ARRAY_FIELD: dict[str, tuple[str, callable]] = {
    'status': ('status', lambda v: _encode(str(v), STATUS_CODES)),
    'privilege': ('privilege', lambda v: _encode(str(v), PRIVILEGE_CODES)),
    'decoy': ('decoy', lambda v: _encode(str(v), DECOY_CODES)),
    'edr_active': ('edr_active', bool),
    'is_domain_controller': ('is_domain_controller', bool),
    'contains_honeytokens': ('contains_honeytokens', bool),
    'human_vulnerability_score': ('human_vulnerability', float),
    'cvss_score': ('cvss_score', float),
}

# Attribute → HostMeta field, for variable-length / string fields.
_META_FIELD = {
    'os': 'os',
    'services': 'services',
    'vulnerabilities': 'vulnerabilities',
    'cached_credentials': 'cached_credentials',
    'system_tokens': 'system_tokens',
}


def _set_host_array(
    state: EnvState, idx: int, field_name: str, encoded_value
) -> EnvState:
    """Return a new EnvState with ``state.hosts.<field>[idx] = encoded_value``."""
    arr = getattr(state.hosts, field_name).copy()
    arr[idx] = encoded_value
    new_hosts = replace(state.hosts, **{field_name: arr})
    return replace(state, hosts=new_hosts)


def _set_host_meta(
    state: EnvState, idx: int, field_name: str, value
) -> EnvState:
    current = list(getattr(state.meta, field_name))
    if field_name in ('services', 'vulnerabilities', 'cached_credentials', 'system_tokens'):
        current[idx] = tuple(value) if not isinstance(value, str) else (value,)
    else:
        current[idx] = value
    new_meta = replace(state.meta, **{field_name: tuple(current)})
    return replace(state, meta=new_meta)


def _set_compromised_by(state: EnvState, idx: int, agent_id: str) -> EnvState:
    if agent_id == 'None' or agent_id is None:
        code = -1
    elif agent_id in state.agent_ids:
        code = state.agent_ids.index(agent_id)
    else:
        # Agent not in canonical list — store as -1 and let to_global_state
        # surface it as 'None'. Documented limitation; widen agent_ids if
        # this becomes load-bearing.
        code = -1
    arr = state.hosts.compromised_by_id.copy()
    arr[idx] = code
    new_hosts = replace(state.hosts, compromised_by_id=arr)
    return replace(state, hosts=new_hosts)


def apply_state_delta(state: EnvState, delta_key: str, delta_value=None) -> EnvState:
    """Pure interpreter for legacy ``state_deltas`` entries.

    Returns a new :class:`EnvState`. Unknown keys are ignored to match the
    legacy behavior (the legacy ``apply_delta`` silently no-ops on
    unrecognized attribute names via the ``hasattr`` guard).
    """
    if not isinstance(delta_key, str):
        # Command-object deltas are out of scope for this slice — see module
        # docstring above.
        return state

    parts = delta_key.split('/')

    if parts[0] == 'hosts' and len(parts) == 3:
        ip, attribute = parts[1], parts[2]
        if ip not in state.meta.ip:
            return state
        idx = state.meta.ip.index(ip)

        if attribute == 'compromised_by':
            return _set_compromised_by(state, idx, delta_value)
        if attribute in _ARRAY_FIELD:
            field_name, encoder = _ARRAY_FIELD[attribute]
            return _set_host_array(state, idx, field_name, encoder(delta_value))
        if attribute in _META_FIELD:
            return _set_host_meta(state, idx, _META_FIELD[attribute], delta_value)
        # Unknown host attribute — legacy silently ignores.
        return state

    if parts[0] == 'knowledge' and len(parts) == 3:
        agent_id, ip = parts[1], parts[2]
        if agent_id not in state.agent_ids:
            return state
        j = state.agent_ids.index(agent_id)
        new_set = state.knowledge[j] | {ip}
        new_knowledge = tuple(
            new_set if k == j else s for k, s in enumerate(state.knowledge)
        )
        return replace(state, knowledge=new_knowledge)

    # 'history/...' and 'firewall/...' are not yet modeled in EnvState.
    return state


def apply_state_deltas(state: EnvState, deltas) -> EnvState:
    """Apply a dict or list of deltas left-to-right. Mirrors the two shapes
    the legacy env feeds into ``apply_delta`` (see parallel_env.py:463-469).
    """
    if isinstance(deltas, dict):
        for k, v in deltas.items():
            state = apply_state_delta(state, k, v)
    elif isinstance(deltas, (list, tuple)):
        for item in deltas:
            # Legacy list form: each entry is itself a delta key (command
            # object or string) — preserve that contract.
            state = apply_state_delta(state, item)
    return state


# ── Pure conflict resolution ───────────────────────────────────────────────
#
# Functional companion to ``ConflictResolutionEngine.resolve``. Same
# semantics ("Blue defensive supremacy on simultaneous same-target hits")
# but does NOT mutate input ActionEffects — returns a fresh dict whose
# values are either the original effect or a new ActionEffect with success
# nullified. Required for the JAX backend, where any in-place mutation in a
# traced function silently breaks correctness.


def _extract_targeted_ips(state_deltas) -> set[str]:
    """Pull every ``hosts/<ip>/...`` IP out of a state_deltas payload."""
    ips: set[str] = set()
    if isinstance(state_deltas, dict):
        for key in state_deltas.keys():
            if isinstance(key, str) and key.startswith('hosts/'):
                parts = key.split('/')
                if len(parts) >= 2:
                    ips.add(parts[1])
    elif isinstance(state_deltas, (list, tuple)):
        for delta_obj in state_deltas:
            target = getattr(delta_obj, 'target_ip', None)
            if target:
                ips.add(target)
    return ips


def resolve_conflicts(effects):
    """Pure variant of :meth:`ConflictResolutionEngine.resolve`.

    Returns a NEW dict mapping agent_id → ActionEffect, with Red effects
    nullified if they target a host that any Blue effect simultaneously
    succeeds on. The input dict and its ActionEffects are NOT mutated.

    Behavior mirrors the legacy engine bit-for-bit; the only difference is
    immutability — callers that previously read ``effects[red_id].success``
    after :meth:`resolve` should now read the return value.
    """
    from netforge_rl.core.action import ActionEffect

    blue_defended: set[str] = set()
    for agent_id, eff in effects.items():
        if eff is None or not eff.success:
            continue
        if 'blue' in agent_id.lower():
            blue_defended |= _extract_targeted_ips(eff.state_deltas)

    resolved = {}
    for agent_id, eff in effects.items():
        if eff is None or not eff.success or 'red' not in agent_id.lower():
            resolved[agent_id] = eff
            continue

        red_targets = _extract_targeted_ips(eff.state_deltas)
        if red_targets & blue_defended:
            empty_deltas = [] if isinstance(eff.state_deltas, list) else {}
            new_obs = dict(eff.observation_data)
            new_obs['alert'] = 'TEMPORAL_COLLISION_DEFENSE_SUPREMACY'
            resolved[agent_id] = ActionEffect(
                success=False,
                state_deltas=empty_deltas,
                observation_data=new_obs,
                eta=eff.eta,
                action=eff.action,
            )
        else:
            resolved[agent_id] = eff

    return resolved
