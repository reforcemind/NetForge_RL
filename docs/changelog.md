# Changelog

All notable changes to the `netforge_rl` project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [3.1.0] — 2026-07-10

### Fixed
- **JAX backend action timing** (`backends/jax/transition.py`, `backends/jax/action_codes.py`):
  the vectorized backend now enqueues and resolves actions against the same per-agent duration
  table as the real Python action classes, instead of resolving every action within the tick
  it's submitted. An agent cannot submit a new action while one is still pending (mirroring
  `parallel_env.py`'s `agent_locked_until`), and a Blue `IsolateHost` that matures on a tick now
  cancels any still-pending Red action on the same host, with same-tick ties favoring Blue.
  Every one of the 34 action durations was re-derived from its actual Python action class
  rather than assumed; a prior, unfinished attempt at this had 10 of 34 durations wrong, a
  deleted `scenario_done`, a reference to a nonexistent `hosts.reachability` field, a
  `StepEvents` construction using stale field names, and a broken `siem_buffer` reference in
  the JaxMARL bridge.
- **`backends/reference.py`**: the NumPy mirror used for JAX parity testing carries the same
  queue logic, verified against the JAX kernel with zero mismatches across all five scenarios
  under randomized rollouts, plus targeted deterministic checks of the isolation-cancellation
  and submission-lock mechanics.

### Removed
- `backends/jax/siem_embeddings.py`: an unfinished, untested SIEM-embedding-lookup attempt
  that referenced an unimported `LogEncoder` name and raised `NameError` on first use. The
  existing `jax_siem_features` scalar signal remains the JAX backend's telemetry proxy; the
  JAX and Python backends still diverge on observation richness and on tick cadence (the JAX
  core advances a fixed one tick per `step()` call rather than jumping to the next event).

## [3.0.0] — 2026-07

### Added
- **Trained baselines + curves**: `benchmarks/train_curve.py` runs the JIT-fused JAX IPPO
  trainer, records the reward/loss curves, renders a chart, and checkpoints the policy
  (`save_params`/`load_params`). A committed 40-iter `ransomware` run learns 0.06 → 0.71 mean
  reward over 245,760 env-steps.
- **Gymnasium single-agent env** (`environment/gym_env.py`): `NetForgeSingleAgentEnv` controls
  one agent against scripted opponents, passes `gymnasium.check_env`, and exposes the action
  mask in `info` — a drop-in target for Stable-Baselines3 / CleanRL.
- **MITRE ATT&CK coverage** (`actions/attack_map.py`): red actions map to ATT&CK techniques;
  episodes report `attack_techniques` and `attack_coverage` in `info`.
- **Graph-native observations** (`core/graph_obs.py`, `environment/graph_wrapper.py`):
  node/edge/edge-attr arrays with fog-of-war masking, one call to PyTorch Geometric via
  `to_pyg`; `GraphObservationWrapper` injects them into `info` without changing obs shapes.
- **Capability cards** (`diagnostics/capability_card.py`): run the 6-probe suite across seeds
  and emit a per-policy JSON + radar chart.
- **Deception mechanics**: `deception_hits` / `deception_efficacy` metrics measuring how much
  of Red's effort landed on decoys and honeytokens.
- **SOC export** (`siem/export.py`): capture the full episode SIEM stream (`record_siem=True`)
  and export it as OCSF-style JSON records for real SIEM tooling.
- **Self-play & Elo** (`benchmarks/self_play.py`): a population tournament that rates red and
  blue policies on one SLA-based ladder.
- **JAX-native SIEM signal** (`backends/jax/vector_env.py:jax_siem_features`): a per-host
  numeric alert vector computed in-kernel, wired into the vectorized blue observation behind
  `JaxMARLEnv(telemetry_obs=True)` so the fast backend carries telemetry end-to-end.

### Fixed
- **Packaging**: `scikit-learn` and `pillow` are hard runtime dependencies (the default
  `LogEncoder` tfidf backend and the LLM vision prompt builder both import them
  unconditionally) but neither was in core `dependencies` — a base `pip install netforge_rl`
  could not import the environment or the semantic package. Both moved to core deps.
- **Circular import**: `netforge_rl.baselines.eval` imported `EpisodeResult` from the
  `netforge_rl.semantic` package root, which re-enters `baselines` via `semantic.modes`
  during its own init. Imports now point at the leaf modules (`semantic.runner`,
  `semantic.leaderboard`) that don't import `baselines`, breaking the cycle with no
  lazy/in-function imports.
- **Tests**: two JAX-dependent test files imported `jax` directly instead of the repo's
  established `pytest.importorskip('jax')` guard, so they errored (rather than skipped)
  in environments without the `jax` extra installed (e.g. CI).

### Removed
- `benchmarks/sps_baseline.py`, `sps_jax_vectorized.py`: superseded by
  `benchmarks/throughput.py`, which the docs already pointed at.
- `benchmarks/train_ippo.py`: superseded by `benchmarks/train_curve.py`.
- `benchmarks/baseline_sweep.py`: an unused, undocumented, weaker duplicate of
  `run_benchmark.py` + `build_leaderboard.py`.

### Docs
- Rewrote `README.md` and `docs/index.md` for the v3.0.0 feature set.
- Removed duplicated throughput/competition/scoring/sweep instructions that were
  repeated across `benchmarks/overview.md`, `baselines.md`, and `run.md`; each page now
  owns one topic and links to the others instead of restating them.

## [2.3.0] — 2026-07

### Added
- **Difficulty presets**: `netforge_rl.environment.presets` exposes named `easy`/`medium`/
  `hard` tiers (`make_config`, `make_env`) and a frozen 20-seed held-out `EVAL_SEEDS`
  suite, so difficulty and the train/eval split are reproducible and comparable.
- **Config knobs wired**: `log_latency` now actually delays SIEM log visibility (a lagging
  SOC feed; 0 = immediate, the historical behaviour) and `dhcp_interval` now controls DHCP
  churn (previously hard-coded to 40 and ignored from config).
- **Diagnostics**: expanded the probe suite from 2 to 6 capabilities — added
  `DelayedTelemetry` (temporal), `FalsePositiveRestraint` (precision), `OTKineticResponse`
  (safety), and `TopologyShift` (generalization) alongside `MemoryProbe` and `NoisySIEM`.
- **Baselines**: `KillChainRedPolicy`, a scripted recon→exploit→pivot attacker that
  actually compromises hosts (the naive `HeuristicRedPolicy` skipped recon, so its
  exploits always failed the prior-state check and nothing was ever compromised).
  Wired into `run_benchmark`/`build_leaderboard`; the regenerated leaderboards now show
  non-zero, CI-bounded compromise rates.
- **Spec**: `REWARD_WEIGHTS` tables on each scenario and `get_reward_weights()`, so
  `benchmarks/env_spec.py` publishes a stable reward-decomposition spec.

### Docs
- Rewrote the site to match the current environment: refreshed the landing page and
  quickstart; added pages for **Difficulty & Splits**, **Reproducibility**, the
  **Diagnostic** suite, and **Baselines**; corrected throughput claims to measured
  env-steps/s vs agent-steps/s; and expanded the datasheet. All code snippets are
  smoke-tested and the site builds under `mkdocs --strict`.

### Fixed
- **Imports**: `netforge_rl.baselines.eval` now imports `netforge_rl.semantic` lazily,
  breaking a `semantic → modes → baselines → eval → semantic` import cycle.
- **Determinism**: The sim `MockHypervisor` RNG was seeded once at construction and
  never reset, so exploit outcomes leaked across episodes. It is now reseeded on
  `reset(seed)`, closing a latent reproducibility hole in the exploit path.
- **Determinism**: SIEM event templates now take a per-call RNG (threaded from the
  env/logger) instead of a shared module global, so concurrent envs in one process
  stay independent and reproducible.
- **Benchmarks**: Fixed `env_spec.py` (broken `get_reward_weights` import) and
  `build_leaderboard` (broken `evaluate` import); baseline evaluation excludes
  `169.254.0.0/16` padding from compromised/isolated counts.
- **Rewards**: Added `iter_host_deltas` so reward, metric, and info code reads host
  changes identically whether an action returns a dict delta or a command-list delta.
  Command-based actions (e.g. `ExploitRemoteService`, `SpearPhishing`) are now credited
  for privilege/isolation changes they were previously ignored for.
- **Actions**: Unified the action taxonomy under one team per role (`red`/`blue`).
  Blue detection actions (`Monitor`, `Analyze`, `DeployEDR`, decoys, honeytokens,
  `RotateKerberos`) are now reachable by the default blue agents; the `operator`/
  `commander` split that no live agent used was removed.
- **Actions**: `DumpLSASS` and `RotateKerberos` returned their command wrapped in a
  string-keyed dict, so it never executed; they now return command lists.
- **Determinism**: Stochastic actions draw from a per-episode RNG on
  `GlobalNetworkState`; the SIEM logger and event templates are reseeded on
  `reset(seed)` and timestamps are derived from a fixed epoch. Rewards and telemetry
  now replay identically under a seed.
- **Metrics**: `169.254.0.0/16` padding hosts are excluded from compromised/isolated/
  SLA metrics; a separate `active_hosts` count is reported.
- **Scenarios**: Fixed the OT subnet name mismatch (`OT` vs `OT_Subnet`) so OT-specific
  reward branches fire.
- **Benchmarks**: `python -m benchmarks.run_benchmark` runs the sweep on the default
  path (previously a no-op unless `--gap`); heuristic policies are now driven through
  an env-bound `PolicyAgent` instead of silently falling back to random.

## [2.2.0] — 2026-06

### Fixed
- **Core**: Normalized rewards using `tanh` scaling to prevent reward explosion.
- **Core**: Removed module-level global seeding for deterministic parallel execution.
- **Core**: Fixed `red` agent action mapping and removed duplicate conflict resolution logic.
- **Core**: Fixed fog-of-war masking shapes to align with the 132-dimension observation space.
- **Core**: Replaced hardcoded decoy IPs with dynamic `169.254.x.x` padding IPs in reconnaissance.

---

## [2.1.0] — 2026-06

### Added
- **Environment**: Added `TopologyEventEngine` to `netforge_rl/environment/parallel_env.py` to support dynamic topologies.
- **Environment**: Implemented curriculum learning framework.
- **Training**: Added RLlib integration and RMAPPO vs RMAPPO baseline.

### Fixed
- **Core**: Fixed decoy recon replacing hosts, static action mask issues, and stale DHCP targeting.

---

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
