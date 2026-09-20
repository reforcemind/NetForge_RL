from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from netforge_rl.datasets.collector import Trajectory

MINARI_SPEC = '0.5'
DATASET_ID_PREFIX = 'netforge'


def trajectories_to_minari_dict(
    trajectories: Sequence[Trajectory],
    dataset_id: str = 'netforge/arena-mixed-v0',
) -> dict:
    """Minari-compatible episode dict (no hard dependency on the minari package)."""
    episodes = []
    for traj in trajectories:
        agents = sorted({a for step in traj.rewards for a in step})
        agent = agents[0] if agents else 'blue_dmz'
        obs = np.array(
            [
                step.get(agent, {}).get('obs', np.zeros(256, dtype=np.float32))
                for step in traj.observations
            ],
            dtype=np.float32,
        )
        acts = np.array(
            [step.get(agent, [0, 0]) for step in traj.actions], dtype=np.int64
        )
        rews = np.array(
            [step.get(agent, 0.0) for step in traj.rewards], dtype=np.float32
        )
        terms = np.array(
            [step.get(agent, False) for step in traj.terminals], dtype=np.bool_
        )
        truncs = np.array(
            [step.get(agent, False) for step in traj.truncations], dtype=np.bool_
        )
        episodes.append(
            {
                'observations': obs,
                'actions': acts,
                'rewards': rews,
                'terminations': terms,
                'truncations': truncs,
                'seed': traj.seed,
                'scenario': traj.scenario,
                'quality': traj.policy,
            }
        )
    return {
        'id': dataset_id,
        'minari_version': MINARI_SPEC,
        'env_spec': 'netforge_rl/NetForge-v4',
        'n_episodes': len(episodes),
        'episodes': episodes,
        'notes': (
            'Offline-RL export. Flattened Blue obs for the first controlled agent. '
            'Publish to Hugging Face as parquet/npz; load with minari if installed.'
        ),
    }


def export_npz(trajectories: Sequence[Trajectory], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = trajectories_to_minari_dict(trajectories)
    arrays = {}
    meta_episodes = []
    for i, ep in enumerate(payload['episodes']):
        arrays[f'ep{i}_obs'] = ep['observations']
        arrays[f'ep{i}_act'] = ep['actions']
        arrays[f'ep{i}_rew'] = ep['rewards']
        arrays[f'ep{i}_term'] = ep['terminations']
        arrays[f'ep{i}_trunc'] = ep['truncations']
        meta_episodes.append(
            {
                'seed': ep['seed'],
                'scenario': ep['scenario'],
                'quality': ep['quality'],
                'length': len(ep['rewards']),
            }
        )
    meta = {
        k: payload[k]
        for k in ('id', 'minari_version', 'env_spec', 'n_episodes', 'notes')
    }
    meta['episodes'] = meta_episodes
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def export_hdf5(trajectories: Sequence[Trajectory], path: str | Path) -> Path:
    """Minari-like HDF5 (episode groups). Requires optional ``h5py`` / ``offline`` extra."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            'h5py is optional. pip install "netforge-rl[offline]" or use export_npz().'
        ) from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = trajectories_to_minari_dict(trajectories)
    with h5py.File(path, 'w') as handle:
        handle.attrs['id'] = payload['id']
        handle.attrs['minari_version'] = payload['minari_version']
        handle.attrs['env_spec'] = payload['env_spec']
        handle.attrs['n_episodes'] = payload['n_episodes']
        handle.attrs['notes'] = payload['notes']
        for i, ep in enumerate(payload['episodes']):
            group = handle.create_group(f'episode_{i}')
            group.create_dataset('observations', data=np.asarray(ep['observations']))
            group.create_dataset('actions', data=np.asarray(ep['actions']))
            group.create_dataset('rewards', data=np.asarray(ep['rewards']))
            group.create_dataset('terminations', data=np.asarray(ep['terminations']))
            group.create_dataset('truncations', data=np.asarray(ep['truncations']))
            group.attrs['seed'] = int(ep['seed'])
            group.attrs['scenario'] = str(ep['scenario'])
            group.attrs['quality'] = str(ep['quality'])
    return path


def try_minari_export(
    trajectories: Sequence[Trajectory],
    dataset_id: str,
    path: str | Path | None = None,
) -> Path:
    """Write HDF5 if h5py is present; otherwise npz. Full Minari create is optional."""
    dest = (
        Path(path) if path is not None else Path(f'{dataset_id.replace("/", "_")}.hdf5')
    )
    try:
        return export_hdf5(trajectories, dest)
    except ImportError:
        npz_path = dest.with_suffix('.npz')
        return export_npz(trajectories, npz_path)
