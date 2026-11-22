# Belief-State Graphs

Standard cyber simulators often grant defensive agents complete, privileged views of the network graph. In NetForge, Blue defenders observe reconstructed belief-state graphs rather than ground-truth compromise maps.

## Belief Graph vs. Oracle Graph

| Graph Mode | Intended Agent | Visibility Scope | Data Source |
|---|---|---|---|
| **Belief Graph** | Blue Defender | Host inventory, known network edges, and confirmed SIEM alert states | Reconstructed from telemetry logs and asset scans |
| **Fog-of-War Graph** | Red Adversary | Subnets, hosts, and services explicitly discovered via recon | Built dynamically during kill-chain progression |
| **Oracle Graph** | Diagnostics Only | Ground-truth compromise state, root shells, and active malware | Privileged simulation state (never exposed during policy training) |

## Using the Graph Observation Wrapper

Attach `GraphObservationWrapper` to inject graph-structured dictionaries into `info`:

```python
from pettingzoo import make
from netforge_rl.environment.graph_wrapper import GraphObservationWrapper

raw_env = make("parallel", "netforge/ransomware-v4", max_ticks=80)
env = GraphObservationWrapper(raw_env)

obs, infos = env.reset(seed=42)

# Blue observes the reconstructed belief graph
blue_graph = infos["blue_dmz"]["graph"]
print("Node features shape:", blue_graph["node_features"].shape)
print("Edge index shape:", blue_graph["edge_index"].shape)
```

The underlying PettingZoo observation vector remains standard and flat, ensuring backwards compatibility with non-graph RL algorithms.
