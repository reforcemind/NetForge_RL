# Simulation Reproducibility

Scientific evaluation requires deterministic reproducibility across operating systems and execution runs. NetForge enforces determinism across all simulation components.

## Deterministic Seeding

Calling `env.reset(seed=S)` completely seeds:

- Initial network topology, subnet routing tables, and host vulnerability profiles.
- Adversary exploit sequences and randomized reconnaissance probes.
- Continuous physical simulation noise in OT/SCADA systems.
- SIEM log arrival latency and telemetry drop rates.

Given identical action sequences, two runs from the same seed yield bit-for-bit identical observations, rewards, and info dictionaries.

## Golden Trajectory Verification

NetForge maintains golden trajectory fingerprints in `tests/parity/`:

- `test_seeded_observation_replay.py`: Verifies that replaying stored action sequences against a seeded environment reproduces exact observation vectors.
- `test_golden_trajectory.py`: Compares full episode traces against committed cryptographic hash fingerprints to detect unintended behavioral regressions during development.

## Best Practices for Reproducible Runs

- Always specify an explicit integer seed in `reset(seed=...)`.
- Pin the package version and underlying NumPy version.
- When evaluating, report aggregate performance across at least 20 random seeds alongside confidence intervals or CVaR metrics.
