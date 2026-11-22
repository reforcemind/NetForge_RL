# Dynamic Topology

Real enterprise environments are not static graphs. Workstations enter sleep states, employees roam between subnets, and new endpoints connect to the corporate network.

NetForge models network variability through the `TopologyEventEngine`, which dynamically mutates the environment topology during an episode.

## Topology Events

The engine evaluates stochastic events at each tick:

- **Host Churn (`host_offline` / `host_online`)**: Endpoints temporarily disconnect or power down, terminating active network connections and altering routing paths.
- **Host Migration (`host_migrate`)**: Endpoints change IP addresses or roam between subnets (e.g., transitioning from a wireless corporate network to a remote VPN pool).
- **Host Arrival (`host_arrive`)**: New devices join the network, activated from pre-allocated padding slots in the host array.

## Configuration

Dynamic topology rates can be configured directly in scenario specifications:

```python
from netforge_rl.environment.parallel_env import NetForgeRLEnv

config = {
    "scenario_type": "enterprise_apt",
    "topology_churn_rate": 0.02,        # Probability of host state flip per tick
    "topology_migration_rate": 0.01,    # Probability of subnet migration per tick
    "topology_arrival_rate": 0.005,     # Probability of new host arrival per tick
}

env = NetForgeRLEnv(config)
```

## Curriculum Integration

In curriculum learning setups, dynamic topology events introduce progressive complexity:

- **Novice Phase**: Static network topology with fixed routing.
- **Intermediate Phase**: Periodic DHCP IP reassignments.
- **Expert Phase**: Active churn, roaming subnet migration, and unexpected device arrivals.
