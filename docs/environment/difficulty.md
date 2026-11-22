# Difficulty & Evaluation Splits

NetForge provides difficulty presets and deterministic evaluation splits to benchmark policies under controlled complexity levels.

## Difficulty Presets

Difficulty can be set in the environment configuration:

```python
from netforge_rl.environment.parallel_env import NetForgeRLEnv

env = NetForgeRLEnv({"scenario_type": "enterprise_apt", "difficulty": "medium"})
```

| Difficulty | Network Scale | Adversary Skill | Telemetry Latency | Benign Log Noise |
|---|---|---|---|---|
| **`easy`** | 10–15 hosts | Naive opportunistic scanning; slower exploit cadence | 0 ticks (instant logs) | Low |
| **`medium`** | 20–35 hosts | Multi-stage kill-chain tactics; service reconnaissance | 1–2 ticks | Moderate |
| **`hard`** | 50–100 hosts | Stealthy credential abuse; zero-day exploits; dynamic churn | 2–5 ticks | High (frequent background noise) |

## Evaluation Splits & Seeds

To prevent policies from memorizing fixed host indices or deterministic exploit sequences, evaluations are partitioned into structured seed spaces:

- **`train`**: Open seed range with randomized initial host vulnerabilities and IP assignments.
- **`dev`**: Frozen seed set (`EVAL_SEEDS`) with deterministic host layouts for validation and checkpoint selection.
- **`hidden`**: Held-out seed block reserved for formal competition rankings.

When `evaluation_mode=True` is enabled, the environment activates frozen validation parameters and sets deterministic initial conditions.
