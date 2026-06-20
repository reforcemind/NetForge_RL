# Changelog

All notable changes to the `netforge_rl` project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.0.0] — 2026-06

### Added
- **Core**: Added `netforge_rl/core/functional.py` implementing a pure `EnvState` interpreter.
- **JAX Backend**: Added `netforge_rl/backends/jax/` providing vectorized kernels and batched step execution for high-throughput training.
- **Bridges**: Added `JaxMARL` environment adapter and `dlpack` support for zero-copy array conversion.
- **Rendering**: Implemented decoupled rendering pipeline in `netforge_rl/render/`.
- **Baselines**: Added heuristic evaluation policies and a PPO implementation for the JaxMARL backend.
- **Semantic Interface**: Added `netforge_rl/semantic/` for language-model-based agent interaction and LLM fine-tuning.
- **Notebooks**: Added 7 reference notebooks for quickstart, visualization, and RL training.
- **Tests**: Expanded test suite to 173 tests, including deterministic trajectory hashing.

### Changed
- Increased maximum parallel throughput to >1M steps-per-second via JAX vectorization.

---

## [1.0.0] — 2026-02

### Added
- Native integration with the PettingZoo `ParallelEnv` API.
- Replaced legacy observation spaces with `gymnasium.spaces`.
- Created unified `BaseAction` / `BaseObservation` class hierarchy.
- Upgraded testing and runtime requirements to Python 3.14.

### Removed
- Removed deprecated OpenAI Gym wrappers and legacy evaluation modules.
