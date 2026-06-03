from dataclasses import dataclass

import numpy as np

from netforge_rl.core.functional import (
    DECOY_CODES,
    EnvState,
    PRIVILEGE_CODES,
    STATUS_CODES,
)


COLOR_SECURE = (0.30, 0.70, 0.35)
COLOR_COMPROMISED = (0.85, 0.20, 0.20)
COLOR_DEFENDED = (0.20, 0.45, 0.85)
COLOR_HONEYTOKEN = (0.95, 0.80, 0.15)
COLOR_ISOLATED = (0.55, 0.55, 0.55)
COLOR_DECOY = (0.65, 0.35, 0.75)
COLOR_PANIC = (0.10, 0.10, 0.10)


@dataclass(frozen=True)
class Snapshot:
    """Renderer-ready view; padding nodes filtered out."""
    labels: tuple
    subnets: tuple
    colors: np.ndarray
    edges: tuple
    tick: int

    @property
    def n_nodes(self):
        return self.colors.shape[0]


def _classify(status_code, priv_code, decoy_code, honeytoken, compromised_by, edr_active):
    if STATUS_CODES[status_code] == 'kernel_panic':
        return COLOR_PANIC
    if STATUS_CODES[status_code] == 'isolated':
        return COLOR_ISOLATED
    if honeytoken:
        return COLOR_HONEYTOKEN
    if DECOY_CODES[decoy_code] != 'inactive':
        return COLOR_DECOY
    if compromised_by >= 0 or PRIVILEGE_CODES[priv_code] in ('User', 'Root'):
        return COLOR_COMPROMISED
    if edr_active:
        return COLOR_DEFENDED
    return COLOR_SECURE


def snapshot_from_envstate(state: EnvState) -> Snapshot:
    """Build a render snapshot from a frozen EnvState (padding hosts filtered)."""
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

    edges = []
    by_subnet = {}
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
