# Python Quickstart

NetForge provides standard [Gymnasium](https://gymnasium.farama.org/) and [PettingZoo](https://pettingzoo.farama.org/) interfaces.

## Environment Registration

Importing `netforge_rl` automatically registers standard environment IDs:

| API | Environment ID | Description |
|---|---|---|
| **PettingZoo Parallel** | `netforge/ransomware-v4` | 1 Red adversary + 3 Blue defensive agents |
| **PettingZoo Parallel** | `netforge/arena-v4` | Standard evaluation environment |
| **PettingZoo AEC** | `netforge/ransomware-v4` | Turn-based agent-environment cycle |
| **Gymnasium** | `NetForge/Blue-v4` | Single-agent Blue defense vs. scripted Red |
| **Gymnasium** | `NetForge-v4` | Alias for `NetForge/Blue-v4` |

## Action Space

Actions are represented by `MultiDiscrete([32, 100])`:

- **Dimension 0 (Action Type)**: Index from 0 to 31 corresponding to specific cyber actions (e.g., `DiscoverNetworkServices`, `ExploitRemoteService`, `IsolateHost`, `RestoreHost`).
- **Dimension 1 (Target Host)**: Index from 0 to 99 corresponding to the target host slot in the network topology.

### Action Masks

To prevent invalid or disallowed moves (e.g., isolating an already isolated host or attacking an undiscovered IP), environments provide an action mask of shape `int8[132]` via `info["action_mask"]`:

```python
# Sample a valid legal action using the action mask
action = env.action_space.sample(mask=info["action_mask"])
```

## Observation Space

Blue observations are constructed from SIEM telemetry and inventory data rather than ground-truth simulator state:

- **Vector Observation (`obs`)**: 256-dimensional numerical vector summarizing local host telemetry, security metrics, and alert counts.
- **SIEM Embedding (`siem_embedding`)**: 128-dimensional embedding of recent Sysmon-style logs.
- **Adjacency Matrix (`adj_matrix`)**: Flattened $100 \times 100$ routing and reachability matrix.
- **Delta Time (`delta_t`)**: Normalized time elapsed since the previous event.
- **Belief-State Graph**: When wrapped with `GraphObservationWrapper`, `info["graph"]` exposes a graph dictionary compatible with PyTorch Geometric (`node_features`, `edge_index`, `edge_attr`).

## Complete Example

```python
import gymnasium as gym
import netforge_rl

env = gym.make("NetForge/Blue-v4", max_ticks=80)
obs, info = env.reset(seed=42)

total_reward = 0.0
done = False

while not done:
    mask = info.get("action_mask")
    action = env.action_space.sample(mask=mask)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode finished. Total reward: {total_reward:.2f}")
print(f"SLA Uptime: {info.get('SLA_Uptime_Percentage', 0):.1f}%")
print(f"Compromised hosts: {info.get('compromised_hosts', 0)}")
env.close()
```

## Optional Extras

| Extra | Description |
|---|---|
| `dev` | Development tools: `ruff`, `pytest` |
| `docs` | Documentation builder: `sphinx`, `furo`, `myst-parser` |
| `jax` | High-throughput vectorized simulation backend |
| `rllib` | Ray RLlib multi-agent environment bridge |
| `offline` | Trajectory logging and HDF5 dataset export |
| `nlp` | Text embeddings for SIEM log parsing |
