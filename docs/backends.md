# Python vs JAX

Python is the gym. JAX is a throughput surrogate — not a SIEM replica.

| | Python | JAX |
|---|---|---|
| Role | Arena, SIEM, probes | massive rollouts |
| SIEM / NLP | yes | scalar alert |
| Durative / event time | yes | reduced core |
| Belief graphs | yes | host arrays (not yet) |
| Parity | golden trajectory | NumPy reference kernel |

`tests/parity/test_jax_reference_equivalence.py`, `test_golden_trajectory.py`.
SIEM / deception / probes → Python. 10⁵ steps/s → JAX, state the subset.
