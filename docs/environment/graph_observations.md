# Belief-state graphs

Blue’s graph is reconstructed from inventory + SIEM. Red’s is fog of war.
The privileged map is for diagnostics, not for training Blue.

| Mode | Who | |
|---|---|---|
| `belief` | Blue (default) | assets + alerts |
| `fog` | Red | discovered hosts |
| `oracle` | diagnostics | true compromise |

`GraphObservationWrapper` sets `info["graph"]`.
`oracle=True` also sets `info["oracle_graph"]`.
GNN policies read `node_features`, `edge_index`, `edge_attr`, `node_mask`.
The PettingZoo `obs` vector is unchanged.
