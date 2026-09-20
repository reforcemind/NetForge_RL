# Datasheet & Responsible Use — NetForge RL

Following *Datasheets for Datasets* (Gebru et al.) adapted for a simulated
benchmark environment. This accompanies a NeurIPS Datasets & Benchmarks submission.

## Motivation

- **Purpose.** NetForge is a research gym and benchmark for autonomous cyber
  defense: train, evaluate, break, and compare Red/Blue agents. The simulator
  is the engine; NetForge Arena is the standardized evaluation contract
  (scenarios, seeds, hidden tests, capability probes, agent cards).
- **Gap addressed.** Existing cyber-RL environments are largely single-agent or
  fixed-topology, and they rarely freeze a benchmark protocol. NetForge provides
  procedural topologies, belief-state graph observations, a train/dev/hidden
  split, scenario packs, risk-sensitive metrics, and a verified JAX throughput
  backend.

## Composition

- **Instances.** Episodes are procedurally generated network topologies (100 hosts,
  3–5 subnets, optional OT subnet) with per-host OS/CVE/service/credential profiles.
- **Generation.** `NetworkGenerator(seed)` is deterministic per seed. Training draws
  from arbitrary seeds; evaluation (`evaluation_mode=True`) draws from a disjoint
  held-out pool (seed offset 1000) never seen during training.
- **No personal data.** All hosts, IPs, credentials, and logs are synthetic. CVE
  identifiers (e.g. MS17-010, CVE-2019-0708) reference public catalog entries used
  only as abstract vulnerability labels; no exploit code is included.

## Collection / Construction

- **Dynamics.** Two backends share one action taxonomy. The full-fidelity PettingZoo
  env (`environment/parallel_env.py`) carries action durations, an event queue, SIEM
  telemetry, and command-list deltas. The vectorized JAX kernel
  (`backends/jax/vector_env.py`) implements a reduced transition core and is verified
  equivalent to its NumPy reference (`backends/reference.py`) by a trajectory-level
  parity test that gates CI (`tests/parity/test_jax_reference_equivalence.py`). The
  JAX kernel is a scalable subset, not a bit-equivalent replica of the full env.
- **Rewards.** Per-scenario decompositions; see `benchmarks/env_spec.py` for the full
  table. Per-step rewards are small; the OT-safety scenarios (`ot_stuxnet`, and the
  kinetic branch of `ransomware`) add a large ±10,000 terminal signal when a PLC is
  physically destroyed, so catastrophic physical outcomes dominate the return.
  `BaseScenario.normalized_reward` exposes a `tanh`-squashed value in `[-1, 1]` for
  cross-scenario comparison.
- **Determinism.** Deterministic under a seed. A per-episode RNG on `GlobalNetworkState`
  drives stochastic actions (exploit rolls, phishing, PLC overload); the SIEM logger,
  event templates, topology engine, physics, and green agent are all reseeded on
  `reset(seed)`. See `tests/parity/test_golden_trajectory.py`.

## Recommended Uses

- Benchmarking cooperative/competitive MARL algorithms under partial observability.
- Evaluating LLM agents as SOC operators (`benchmarks/llm_eval.py`).
- Studying generalization via the train/held-out topology split
  (`benchmarks/run_benchmark.py --gap`).

## Limitations

- **It is a simulation.** Dynamics are abstracted and have **not** been validated
  against real network telemetry or live red-team engagements. Results do not
  transfer to operational systems without further study.
- **Abstracted exploits.** Exploit success is modeled via CVSS-weighted probabilities
  and vulnerability flags, not real exploitation. CVE labels are not a fidelity claim.
- **No real C2 integration.** Live-fire orchestration against real frameworks is
  intentionally out of scope and gated behind an unimplemented isolation policy.

## Responsible Use (dual-use statement)

This environment trains both attacker and defender policies. It is released for
**defensive research and benchmarking**. The simulator contains no operational
exploit code, no malware, and no capability to act on real systems. Trained Red
policies operate only inside the simulator's abstract action space and confer no
real-world offensive capability. Users must not attempt to connect the environment
to real or production infrastructure. We encourage use aligned with coordinated
vulnerability disclosure norms.

## Maintenance

- **Versioning.** Environment changes that alter dynamics are reflected in the
  parity reference and gated by CI; the golden-trajectory fingerprint test detects
  unintended reward/termination drift.
- **Reproducibility.** Pin the package version and seeds; `benchmarks/` scripts emit
  timestamped JSON results with mean ± CI95.
