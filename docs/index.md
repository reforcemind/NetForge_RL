<div align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&cachebust=1" alt="Python 3.14"/>
  <img src="https://img.shields.io/badge/JAX-vmap%20%2B%20jit-orange?style=for-the-badge" alt="JAX"/>
  <img src="https://img.shields.io/badge/PettingZoo-MARL-purple?style=for-the-badge" alt="PettingZoo"/>
  <img src="https://img.shields.io/badge/MITRE_ATT%26CK-Mapped-red?style=for-the-badge" alt="MITRE ATT&CK"/>
  <img src="https://img.shields.io/badge/LLM%20policies-trl%20%2B%20LoRA-green?style=for-the-badge" alt="LLM policies"/>
</div>

<h1 align="center">NetForge RL</h1>

<p align="center">
  <b>Multi-agent cybersecurity environment for RL research. Red team vs Blue team on generated networks.
</b>
</p>

---

## Features (v2.0)

- **JAX Backend**: Vectorized step implementation (`jax.vmap`, `jax.jit`) providing >1,000,000 steps-per-second at 4096 parallel environments on CPU.
- **JaxMARL & DLPack Integration**: Native integration with hardware-accelerated MARL frameworks and zero-copy tensor conversion between JAX and PyTorch.
- **Decoupled Rendering**: Matplotlib + NetworkX visualizer operating strictly off the hot path, with `moviepy` frame recording capabilities.
- **Semantic Bridge**: Foundation model interfaces providing SIEM telemetry via `state_to_text`, multimodal payloads via `build_vla_prompt`, and LoRA+PPO fine-tuning recipes for Llama-3-8B.
- **Functional Core**: Immutable `EnvState` PyTree with deterministic `apply_state_delta` execution.

## Quick Start

```bash
pip install 'netforge_rl[jax,render,finetune] @ git+https://github.com/reforcemind/NetForge_RL'
```

See the [Quick Start Guide](quickstart.md) for execution loops.

## Architecture Structure

```
netforge_rl/
├── core/               Immutable EnvState PyTree and interpreters
├── backends/jax/       Vectorized kernels and batched execution
├── environment/        Legacy PettingZoo backend
├── bridges/            JaxMARL adapter and zero-copy dlpack
├── render/             Decoupled visualization pipelines
├── semantic/           Interfaces for language model agents
├── baselines/          Reference heuristic policies and JAX PPO
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
