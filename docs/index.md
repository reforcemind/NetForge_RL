<div align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&cachebust=1" alt="Python 3.14"/>
</div>

# NetForge RL

Multi-agent cybersecurity environment for RL research.

---

## Features (v2.2.0)

- **JAX Backend**: Vectorized execution via `jax.vmap` and `jax.jit`.
- **JaxMARL Integration**: Compatible with PyTorch and MARL frameworks.
- **Rendering**: Matplotlib visualization and `moviepy` recording.
- **Language Models**: Utilities for text-based telemetry and fine-tuning.
- **Core**: Immutable `EnvState` with deterministic execution.

## Quick Start

```bash
pip install 'netforge_rl[jax,render,finetune] @ git+https://github.com/reforcemind/NetForge_RL'
```

See the [Quick Start Guide](quickstart.md) for execution loops.

## Architecture Structure

```
netforge_rl/
├── core/               EnvState PyTree and interpreters
├── backends/jax/       Vectorized kernels
├── environment/        PettingZoo backend
├── bridges/            JaxMARL adapter and dlpack
├── render/             Visualization pipelines
├── semantic/           Language model agent interfaces
├── baselines/          Heuristic policies and JAX PPO
└── scenarios/          Environment goal configurations
```

## Citation

```bibtex
@misc{netforge_rl_2026,
  title  = {NetForge RL},
  author = {ReforceMind},
  year   = {2026},
  url    = {https://github.com/reforcemind/NetForge_RL}
}
```
