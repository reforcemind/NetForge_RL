from __future__ import annotations

from typing import Optional

import numpy as np

from netforge_rl.siem.pcap_synthesizer import NODE_DIM, _PRIV_ENC, _SUBNET_ENC

MAX_NODES = 100


def build_graph_observation(
    global_state,
    agent_id: Optional[str] = None,
    max_nodes: int = MAX_NODES,
) -> dict:
    """Graph view of the network (node_features, edge_index, edge_attr, node_mask),
    mapping onto PyTorch Geometric / jraph. Red agents see only discovered hosts."""
    sorted_ips = sorted(global_state.all_hosts.keys())[:max_nodes]
    n = len(sorted_ips)
    idx = {ip: i for i, ip in enumerate(sorted_ips)}

    is_red = bool(agent_id) and 'red' in agent_id.lower()
    known = global_state.agent_knowledge.get(agent_id, set()) if is_red else None

    node_features = np.zeros((max_nodes, NODE_DIM), dtype=np.float32)
    node_mask = np.zeros((max_nodes,), dtype=np.float32)
    for ip, i in idx.items():
        h = global_state.all_hosts[ip]
        if ip.startswith('169.254.'):
            continue
        if is_red and ip not in known:
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
            cross = (
                global_state.get_subnet_name(host.subnet_cidr)
                != global_state.get_subnet_name(global_state.all_hosts[jp].subnet_cidr)
            )
            attr.append(1.0 if cross else 0.0)

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
