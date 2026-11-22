# Simulation Backends

NetForge provides two complementary execution backends: a high-fidelity Python environment and a high-throughput vectorized JAX kernel.

## Backend Comparison

| Feature | Python Backend | Vectorized JAX Backend |
|---|---|---|
| **Primary Use** | Evaluation, SIEM analysis, GNN training | High-throughput policy search & pretraining |
| **Observation** | Full vector + SIEM log buffer + belief graph | Fixed-size array + scalar alert feature |
| **Action Resolution** | Full multi-tick queue with durative locks | Multi-tick queue with kernel-level duration table |
| **Topology** | Dynamic churn, migration, and arrivals | Fixed host array |
| **Throughput (CPU)** | ~300 env-steps / second | ~270,000 env-steps / second |

## Parity Verification

The JAX kernel (`backends/jax/`) is continuously verified against the Python reference kernel (`backends/reference.py`) to ensure identical action timing, queue resolution, and isolation mechanics.

Parity tests are maintained in `tests/parity/`:

- `test_jax_reference_equivalence.py`: Validates deterministic step equivalence across all action codes.
- `test_golden_trajectory.py`: Ensures identical reward trajectories across randomized seeds.
