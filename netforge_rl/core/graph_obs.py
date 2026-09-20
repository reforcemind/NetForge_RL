from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from netforge_rl.siem.pcap_synthesizer import _PRIV_ENC, _SUBNET_ENC, NODE_DIM

MAX_NODES = 100
_IP_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
_SEV_RE = re.compile(r'severity\s*[=:]\s*(\d+)', re.IGNORECASE)

# Belief-feature layout (same NODE_DIM as oracle, different semantics):
# 0 believed_compromised, 1 online, 2 seen_in_siem, 3 unused (no decoy leak),
# 4 unused (no DC leak), 5 subnet prior if inventory, 6 alert_severity/10,
# 7 isolated (Blue's own action / reachable status).
BELIEF_COMPROMISED = 0
BELIEF_ONLINE = 1
BELIEF_SEEN = 2
BELIEF_SUBNET = 5
BELIEF_SEVERITY = 6
BELIEF_ISOLATED = 7


def _iter_siem_logs(global_state) -> Iterable[object]:
    buffer = getattr(global_state, 'siem_log_buffer', None) or []
    current_tick = getattr(global_state, 'current_tick', 0)
    for entry in buffer:
        if isinstance(entry, tuple):
            log = entry[0]
        else:
            log = entry
        if isinstance(log, dict):
            arrival = log.get('arrival_tick', 0) or 0
            if arrival > current_tick:
                continue
        yield log


def _ips_from_log(log) -> list[str]:
    ips: list[str] = []
    if isinstance(log, dict):
        for key in ('target', 'src', 'dst', 'ip', 'source', 'destination'):
            val = log.get(key)
            if isinstance(val, str) and _IP_RE.fullmatch(val.strip()):
                ips.append(val.strip())
        for val in log.values():
            if isinstance(val, str):
                ips.extend(_IP_RE.findall(val))
    else:
        ips.extend(_IP_RE.findall(str(log)))
    return list(dict.fromkeys(ips))


def _severity_from_log(log) -> float:
    if isinstance(log, dict):
        return float(log.get('severity', 0) or 0)
    match = _SEV_RE.search(str(log))
    return float(match.group(1)) if match else 0.0


def siem_belief(global_state) -> dict[str, dict]:
    """Per-host SIEM evidence: mention counts and max severity. No ground truth."""
    belief: dict[str, dict] = {}
    for log in _iter_siem_logs(global_state):
        sev = _severity_from_log(log)
        for ip in _ips_from_log(log):
            slot = belief.setdefault(ip, {'mentions': 0, 'severity': 0.0})
            slot['mentions'] += 1
            slot['severity'] = max(slot['severity'], sev)
    return belief


def resolve_graph_mode(agent_id: str | None, mode: str) -> str:
    """Map ``auto`` onto fog / belief / oracle from the agent role."""
    if mode != 'auto':
        return mode
    if not agent_id:
        return 'oracle'
    lowered = agent_id.lower()
    if 'red' in lowered:
        return 'fog'
    if 'blue' in lowered:
        return 'belief'
    return 'oracle'


def build_graph_observation(
    global_state,
    agent_id: str | None = None,
    max_nodes: int = MAX_NODES,
    mode: str = 'auto',
) -> dict:
    """Graph view of the network (node_features, edge_index, edge_attr, node_mask).

    Modes:
      * ``oracle`` — privileged simulator state (diagnostics / replay only).
      * ``fog`` — Red recon knowledge.
      * ``belief`` — Blue inventory + SIEM-reconstructed compromise/edges.
      * ``auto`` — fog for Red, belief for Blue, oracle if no agent is given.
    """
    resolved = resolve_graph_mode(agent_id, mode)
    if resolved == 'belief':
        return _belief_graph(global_state, agent_id, max_nodes)
    if resolved == 'fog':
        return _oracle_or_fog_graph(global_state, agent_id, max_nodes, fog=True)
    return _oracle_or_fog_graph(global_state, agent_id, max_nodes, fog=False)


def build_oracle_graph(global_state, max_nodes: int = MAX_NODES) -> dict:
    """Privileged graph. Keep out of the Blue observation; diagnostics only."""
    return _oracle_or_fog_graph(
        global_state, agent_id=None, max_nodes=max_nodes, fog=False
    )


def _oracle_or_fog_graph(global_state, agent_id, max_nodes, fog: bool) -> dict:
    sorted_ips = sorted(global_state.all_hosts.keys())[:max_nodes]
    n = len(sorted_ips)
    idx = {ip: i for i, ip in enumerate(sorted_ips)}
    known = (
        global_state.agent_knowledge.get(agent_id, set()) if fog and agent_id else None
    )

    node_features = np.zeros((max_nodes, NODE_DIM), dtype=np.float32)
    node_mask = np.zeros((max_nodes,), dtype=np.float32)
    for ip, i in idx.items():
        h = global_state.all_hosts[ip]
        if ip.startswith('169.254.'):
            continue
        if fog and ip not in known:
            continue
        subnet = global_state.get_subnet_name(h.subnet_cidr)
        node_features[i, 0] = _PRIV_ENC.get(h.privilege, 0.0)
        node_features[i, 1] = 1.0 if h.status == 'online' else 0.0
        node_features[i, 2] = 1.0 if h.compromised_by != 'None' else 0.0
        node_features[i, 3] = 1.0 if h.decoy != 'inactive' else 0.0
        node_features[i, 4] = 1.0 if getattr(h, 'is_domain_controller', False) else 0.0
        node_features[i, 5] = _SUBNET_ENC.get(subnet, 0.0)
        node_features[i, 6] = min(getattr(h, 'cvss_score', 0.0) / 10.0, 1.0)
        node_features[i, 7] = 1.0 if getattr(h, 'edr_active', False) else 0.0
        node_mask[i] = 1.0

    src, dst, attr = [], [], []
    for ip, i in idx.items():
        if node_mask[i] == 0.0:
            continue
        host = global_state.all_hosts[ip]
        for jp, j in idx.items():
            if i == j or node_mask[j] == 0.0:
                continue
            if not global_state.can_route_to(jp, agent_id=agent_id):
                continue
            src.append(i)
            dst.append(j)
            cross = global_state.get_subnet_name(
                host.subnet_cidr
            ) != global_state.get_subnet_name(global_state.all_hosts[jp].subnet_cidr)
            attr.append(1.0 if cross else 0.0)

    return _pack_graph(node_features, node_mask, src, dst, attr, n)


def _belief_graph(global_state, agent_id, max_nodes) -> dict:
    """Uncertain graph: known assets as nodes, compromise/edges from SIEM."""
    sorted_ips = sorted(global_state.all_hosts.keys())[:max_nodes]
    n = len(sorted_ips)
    idx = {ip: i for i, ip in enumerate(sorted_ips)}
    inventory = set()
    if agent_id:
        inventory = set(global_state.agent_knowledge.get(agent_id, set()))
    evidence = siem_belief(global_state)
    telemetry_ips = {ip for ip in evidence if ip in idx}

    visible = set()
    for ip in idx:
        if ip.startswith('169.254.'):
            continue
        if ip in inventory or ip in telemetry_ips:
            visible.add(ip)

    node_features = np.zeros((max_nodes, NODE_DIM), dtype=np.float32)
    node_mask = np.zeros((max_nodes,), dtype=np.float32)
    for ip in visible:
        i = idx[ip]
        host = global_state.all_hosts.get(ip)
        ev = evidence.get(ip, {'mentions': 0, 'severity': 0.0})
        subnet = (
            global_state.get_subnet_name(host.subnet_cidr) if host is not None else ''
        )
        node_features[i, BELIEF_COMPROMISED] = (
            1.0 if ev['severity'] >= 5 or ev['mentions'] >= 2 else 0.0
        )
        node_features[i, BELIEF_ONLINE] = (
            1.0 if host is not None and host.status == 'online' else 0.0
        )
        node_features[i, BELIEF_SEEN] = 1.0 if ev['mentions'] > 0 else 0.0
        node_features[i, BELIEF_SUBNET] = _SUBNET_ENC.get(subnet, 0.0)
        node_features[i, BELIEF_SEVERITY] = min(ev['severity'] / 10.0, 1.0)
        node_features[i, BELIEF_ISOLATED] = (
            1.0 if host is not None and host.status == 'isolated' else 0.0
        )
        node_mask[i] = 1.0

    siem_edges: set[tuple[int, int]] = set()
    for log in _iter_siem_logs(global_state):
        ips = [ip for ip in _ips_from_log(log) if ip in visible]
        for a in ips:
            for b in ips:
                if a != b:
                    siem_edges.add((idx[a], idx[b]))

    src, dst, attr = [], [], []
    visible_list = [ip for ip, i in idx.items() if node_mask[i] == 1.0]
    for ip in visible_list:
        i = idx[ip]
        host = global_state.all_hosts.get(ip)
        for jp in visible_list:
            j = idx[jp]
            if i == j:
                continue
            other = global_state.all_hosts.get(jp)
            same_subnet = (
                host is not None
                and other is not None
                and host.subnet_cidr == other.subnet_cidr
            )
            tele = (i, j) in siem_edges
            if not same_subnet and not tele:
                continue
            src.append(i)
            dst.append(j)
            attr.append(0.0 if same_subnet else 1.0)

    return _pack_graph(node_features, node_mask, src, dst, attr, n)


def _pack_graph(node_features, node_mask, src, dst, attr, n) -> dict:
    edge_index = (
        np.array([src, dst], dtype=np.int64)
        if src
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_attr = (
        np.array(attr, dtype=np.float32).reshape(-1, 1)
        if attr
        else np.zeros((0, 1), dtype=np.float32)
    )
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'node_mask': node_mask,
        'n_nodes': n,
        'n_edges': int(edge_index.shape[1]),
    }


def to_pyg(graph: dict):
    """Convert a graph dict to a PyTorch Geometric ``Data`` object (lazy import)."""
    import torch
    from torch_geometric.data import Data

    return Data(
        x=torch.as_tensor(graph['node_features']),
        edge_index=torch.as_tensor(graph['edge_index']),
        edge_attr=torch.as_tensor(graph['edge_attr']),
    )
