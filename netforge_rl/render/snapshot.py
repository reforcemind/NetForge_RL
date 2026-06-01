"""Pure-CPU view of env state for renderers.

A :class:`Snapshot` is everything a renderer needs and nothing more: host
labels, host status, subnet membership, edges. Built from a frozen
:class:`EnvState` so renderers never reach into mutable backend objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from netforge_rl.core.functional import (
    DECOY_CODES,
    EnvState,
    PRIVILEGE_CODES,
    STATUS_CODES,
)


# Stable color palette by host state. Tuples are RGB in [0, 1].
COLOR_SECURE = (0.30, 0.70, 0.35)       # green   — online, no compromise
COLOR_COMPROMISED = (0.85, 0.20, 0.20)  # red     — Red owns it
COLOR_DEFENDED = (0.20, 0.45, 0.85)     # blue    — Blue elevated / EDR active
COLOR_HONEYTOKEN = (0.95, 0.80, 0.15)   # yellow  — honeytoken trap
COLOR_ISOLATED = (0.55, 0.55, 0.55)     # grey    — pulled offline
COLOR_DECOY = (0.65, 0.35, 0.75)        # purple  — active decoy
COLOR_PANIC = (0.10, 0.10, 0.10)        # black   — kernel panic


@dataclass(frozen=True)
class Snapshot:
    """Renderer-ready view. Active hosts only — padding nodes filtered out."""

    labels: tuple[str, ...]
    subnets: tuple[str, ...]
    colors: np.ndarray            # float[N_ACTIVE, 3]
    edges: tuple[tuple[int, int], ...]
    tick: int

    @property
    def n_nodes(self) -> int:
        return self.colors.shape[0]


def _classify(
    status_code: int,
    priv_code: int,
    decoy_code: int,
    honeytoken: bool,
    compromised_by: int,
    edr_active: bool,
) -> tuple[float, float, float]:
    if STATUS_CODES[status_code] == 'kernel_panic':
        return COLOR_PANIC
    if STATUS_CODES[status_code] == 'isolated':
        return COLOR_ISOLATED
    if honeytoken:
        return COLOR_HONEYTOKEN
    if DECOY_CODES[decoy_code] not in ('inactive',):
        return COLOR_DECOY
    if compromised_by >= 0 or PRIVILEGE_CODES[priv_code] in ('User', 'Root'):
        return COLOR_COMPROMISED
    if edr_active:
        return COLOR_DEFENDED
    return COLOR_SECURE


def snapshot_from_envstate(state: EnvState) -> Snapshot:
    """Build a render snapshot from a frozen EnvState.

    Filters out 169.254.0.0/16 link-local padding hosts — these only exist
    to pin the obs-tensor shape and shouldn't be drawn.
    """
    active_idx = [
        i for i, sn in enumerate(state.meta.subnet_cidr)
        if not sn.startswith('169.254.')
    ]

    labels = tuple(state.meta.hostname[i] for i in active_idx)
    subnets = tuple(state.meta.subnet_cidr[i] for i in active_idx)
    colors = np.array(
        [
            _classify(
                int(state.hosts.status[i]),
                int(state.hosts.privilege[i]),
                int(state.hosts.decoy[i]),
                bool(state.hosts.contains_honeytokens[i]),
                int(state.hosts.compromised_by_id[i]),
                bool(state.hosts.edr_active[i]),
            )
            for i in active_idx
        ],
        dtype=np.float32,
    )

    # Edges: connect every pair of hosts within the same subnet (intra-subnet
    # broadcast domain), plus one inter-subnet uplink between subnet members
    # to keep the layout connected.
    edges: list[tuple[int, int]] = []
    by_subnet: dict[str, list[int]] = {}
    for local_idx, sn in enumerate(subnets):
        by_subnet.setdefault(sn, []).append(local_idx)
    for members in by_subnet.values():
        for a, b in zip(members, members[1:]):
            edges.append((a, b))
    subnet_reps = [members[0] for members in by_subnet.values()]
    for a, b in zip(subnet_reps, subnet_reps[1:]):
        edges.append((a, b))

    return Snapshot(
        labels=labels,
        subnets=subnets,
        colors=colors,
        edges=tuple(edges),
        tick=int(state.current_tick),
    )
