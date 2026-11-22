# Multimodal Observations

NetForge supports multimodal observations combining tabular feature vectors, graph-structured topologies, and simulated packet captures (PCAP).

## Observation Keys

When initialized with multimodal flags (`pcap_obs=True`):

```python
from netforge_rl.environment.parallel_env import NetForgeRLEnv

env = NetForgeRLEnv({"scenario_type": "enterprise_apt", "pcap_obs": True})
obs, infos = env.reset(seed=42)
```

| Observation Key | Shape | Description |
|---|---|---|
| `obs` | `(256,)` | Standard numerical host feature vector and agent telemetry |
| `siem_embedding` | `(128,)` | Encoded representation of recent SIEM log lines |
| `node_features` | `(100, 8)` | Per-host feature matrix for Graph Neural Networks (GNNs) |
| `adj_matrix` | `(10000,)` | Flattened $100 \times 100$ reachability and routing adjacency matrix |
| `pcap` | `(32, 20)` | Sliding snapshot of recent network packets |

## Packet Capture Matrix (PCAP)

The PCAP observation captures 32 recent packet headers across 20 normalized dimensions:

- Dims 0–4: Source host index, destination host index, protocol (TCP/UDP/ICMP), destination port, payload byte size.
- Dims 5–8: TCP flags (`SYN`, `RST`, `ACK`, `PSH`).
- Dims 9–14: Traffic classification flags (lateral movement, C2 beacon, scanning, exfiltration, exploit traffic, sensitive asset target).
- Dims 15–19: Privilege flags, compromise markers, timestamp, encryption flag, severity rating.

Traffic reflects genuine simulation mechanics (command-and-control beacons, network probes, and lateral transfers) rather than synthetic Gaussian noise.

## PyTorch Geometric Integration

Graph observations can be converted directly into PyTorch Geometric `Data` objects:

```python
import torch
from torch_geometric.data import Data

x = torch.tensor(obs["node_features"], dtype=torch.float32)
adj = torch.tensor(obs["adj_matrix"].reshape(100, 100), dtype=torch.float32)

edge_index = adj.nonzero().t().contiguous()
graph_data = Data(x=x, edge_index=edge_index)
```
