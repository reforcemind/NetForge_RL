# Offline RL datasets

Git extra `offline`. Hugging Face dumps are not published yet.

```bash
pip install 'netforge-rl[offline]'
netforge collect --policy heuristic-blue --red killchain-red \
  --episodes 8 --quality mixed --out netforge_offline.npz
netforge collect --hdf5 --out netforge_offline.hdf5
```

`export_npz` / `export_hdf5` are Minari-shaped without importing `minari`.
Mix random / heuristic / RL / self-play / LLM; tag quality before fine-tune.
