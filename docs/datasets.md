# Offline RL Datasets

NetForge supports offline reinforcement learning workflows by logging simulation episodes into standardized trajectory datasets.

## Collecting Trajectories

Install offline dataset utilities:

```bash
pip install -e ".[offline]"
```

Collect trajectories using the CLI:

```bash
# Collect 50 mixed-quality episodes to NumPy NPZ format
netforge collect --policy heuristic-blue --red killchain-red \
  --episodes 50 --quality mixed --out netforge_offline.npz

# Export directly to HDF5 format
netforge collect --policy heuristic-blue --red killchain-red \
  --episodes 50 --hdf5 --out netforge_offline.hdf5
```

## Dataset Schema

Exported datasets follow the [Minari](https://minari.farama.org/) offline dataset structure:

| Field | Shape | Description |
|---|---|---|
| `observations` | `(N, 256)` | Flattened observation vectors across episode transitions |
| `actions` | `(N, 2)` | Action pairs `[action_type, target_host_index]` |
| `rewards` | `(N,)` | Scalar rewards received at each step |
| `terminations` | `(N,)` | Boolean flags indicating terminal state |
| `truncations` | `(N,)` | Boolean flags indicating timeout/truncation |
| `infos/siem_embedding` | `(N, 128)` | Pre-computed embeddings of SIEM log buffers |

Datasets can mix trajectories from random exploration, rule-based heuristics, trained RL checkpoints, and LLM policies with tagged quality labels for behavioral pretraining.
