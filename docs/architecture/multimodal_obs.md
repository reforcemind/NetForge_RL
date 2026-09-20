# Multi-modal obs

```python
env = NetForgeRLEnv({'scenario_type': 'ransomware', 'pcap_obs': True})
```

| Key | Shape | |
|---|---|---|
| `pcap` | (32, 20) | packet snapshot this tick |
| `node_features` | (100, 8) | per-host GNN matrix |
| `adj_matrix` | (10000,) | 100×100 routing (always) |

PCAP dims 0–19: src/dst idx, protocol, port, payload, SYN/RST/ACK/PSH, lateral,
C2, recon, exfil, exploit, dst_sensitive, src_privilege, dst_compromised,
tick_norm, encrypted, severity. All in `[0, 1]`.

Node dims: privilege, online, compromised, decoy, DC, subnet_type, cvss/10, EDR.

Packets come from state (C2 beacons, lateral, then benign fill), not noise.

```python
import torch
from torch_geometric.data import Data

x = torch.tensor(obs['node_features'])
adj = torch.tensor(obs['adj_matrix'].reshape(100, 100))
data = Data(x=x, edge_index=adj.nonzero().t().contiguous())
```
