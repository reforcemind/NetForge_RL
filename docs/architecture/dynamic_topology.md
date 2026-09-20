# Dynamic topologies

`TopologyEventEngine` mutates the graph mid-episode: churn (online/offline),
migration (subnet/IP), arrival (BYOD from padding slots).

Events: `host_offline`, `host_online`, `host_migrate`, `host_arrive`.

```python
cfg = {
    'topology_churn_rate': 0.02,
    'topology_migration_rate': 0.01,
    'topology_arrival_rate': 0.005,
}
env = NetForgeRLEnv(cfg)
```

Rates also scale with {doc}`../training/curriculum`.
