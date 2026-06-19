# Changelog

All notable changes to the `netforge_rl` project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.0.0] — 2026-06 — NeurIPS Datasets & Benchmarks candidate

Dual-backend architecture + Foundation-Model-Native MARL pillar. See [ROADMAP.md](ROADMAP.md) for the full plan; per-phase audit findings in [AUDIT.md](AUDIT.md).

### Added — Phase 0 (audit)
- `benchmarks/sps_baseline.py` — legacy backend SPS baseline (~10 SPS, Win11 / Py 3.12 CPU).
- `tests/parity/test_golden_trajectory.py` — locks SHA-256 fingerprint `abd164a5…` for seed=42, max_ticks=25, ransomware. Every refactor must preserve this hash or re-lock with a changelog entry.
- `AUDIT.md` — inventory, measured numbers, blocker list B1–B8.

### Added — Phase 1 (functional core)
- `netforge_rl/core/functional.py` — frozen `EnvState` PyTree (struct-of-arrays), `HostArrays`, `HostMeta`, codebooks. Pure `apply_state_delta` interpreter mirroring the legacy `apply_delta` shape; pure `resolve_conflicts` companion to `ConflictResolutionEngine.resolve`. Round-trip converters `from_global_state` / `to_global_state` exercised across full episodes (`tests/parity/test_envstate_sync.py`).
- `tests/parity/test_pure_step_equivalence.py` — proves the legacy step and the pure interpreter agree on host arrays across 20 real episode ticks.
- `NetworkGenerator` RNG routed through an explicit `random.Random` instance (resolves blocker B3).

### Added — Phase 2 (JAX backend)
- `netforge_rl/backends/jax/` — registered PyTree `JaxEnvState`, vectorizable kernels (`apply_host_status_delta`, `apply_host_privilege_delta`, `apply_compromised_by_delta`, `resolve_conflicts_mask`), and `vector_env.make_vector_step` — a `jax.vmap`'d, `jax.jit`'d batched step.
- `benchmarks/sps_jax_vectorized.py` — sweep harness.
- **Measured: 1,082,255 aggregate SPS at 4096 envs on CPU** (103,500× the single-env legacy baseline). 4096-env compile verified in CI.

### Added — Phase 3 (API standardization)
- `netforge_rl/bridges/jaxmarl.JaxMARLEnv` — JaxMARL-shape `(obs_dict, state)` / `(obs, state, reward, done, info)` adapter, jit-compatible.
- `netforge_rl/bridges/dlpack` — zero-copy `jax.Array` ↔ `torch.Tensor` via `__dlpack__`.

### Added — Phase 4 (rendering)
- `netforge_rl/render/` — decoupled pipeline (`Snapshot`, `render_rgb`, `FrameRecorder`). `env.render('rgb_array')` returns a uint8 HxWx3 array. Never touches the hot path.

### Added — Phase 5 (baselines)
- `netforge_rl/baselines/policies.py` — `RandomPolicy`, `HeuristicBluePolicy`, `HeuristicRedPolicy` against the legacy env.
- `netforge_rl/baselines/eval.py` — CLI emitting results in the shared leaderboard JSON shape.
- `netforge_rl/baselines/jax_ppo.py` — self-contained PureJaxRL-style PPO (MLP + Adam + GAE + clipped loss), trains on `JaxMARLEnv` without external optax/flax deps.

### Added — Phase 6 (notebooks)
- `notebooks/01_quickstart_pytorch.ipynb`
- `notebooks/02_visualize_attack.ipynb`
- `notebooks/03_train_ppo_cleanrl.ipynb`
- `notebooks/04_mappo_jax_4000envs.ipynb`
- `notebooks/05_sim2real_docker.ipynb`
- `notebooks/06_custom_scenario.ipynb`
- `notebooks/07_finetune_llama3_lora.ipynb` (flagship — Phase 8 M3)

### Added — Phase 8 (Semantic Bridge — Foundation-Model-Native MARL)
- `netforge_rl/semantic/la_wrapper.state_to_text` — frozen EnvState → SIEM-style report, per-role templating, legal action menu + reply format hint.
- `netforge_rl/semantic/vla_wrapper.build_vla_prompt` — pairs RGB frames with text into a provider-neutral payload.
- `netforge_rl/semantic/parser.parse_action` — robust regex parser; `None` on invalid output.
- `netforge_rl/semantic/runner.run_episode` + `leaderboard.{append_result, summarize}` — zero-shot leaderboard harness with `LLMClient` Protocol and a keyless `MockLLMClient` for testing.
- `netforge_rl/semantic/finetune.LMPolicyAdapter` — bridges PettingZoo step onto `trl`'s (query, response, reward) protocol. Reference recipe + 4-bit Llama-3-8B + LoRA config in `finetune/configs/llama3_8b_lora.yaml`.

### Tests
- 173 → from 79 at Phase 0. Golden trajectory hash unchanged across every phase.

---

## [3.0.0] — 2026-02-28
### Added
- **PettingZoo API Core Integration**: `netforge_rl/environment/parallel_env.py` replaces the legacy wrapper paradigm with `pettingzoo.ParallelEnv`.
- **Gymnasium Box Compatibility**: all spaces native to `gymnasium.spaces`.
- **`BaseAction` / `BaseObservation` Hierarchy**: actions return `ActionEffect` payloads that the env resolves under simultaneity.
- **Python 3.12 Support (Native)** via `pyproject.toml`.
- **IPFragmentationAction proof-of-concept**.

### Removed
- **OpenAI Gym legacy layers**: deleted `/Agents/Wrappers/`, `ChallengeWrapper`, `OpenAIGymWrapper`, `TrueTableWrapper`.
- **Demo / dead deps**: legacy `/Evaluation/`, setup.py, legacy `requirements.txt`.
