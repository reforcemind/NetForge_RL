# Diagnostics & Information Asymmetry

NetForge includes diagnostic tools to inspect policy decisions, quantify observability gaps, and analyze hidden network states without leaking privileged information to learning agents.

## Diagnostics Wrapper

The `DiagnosticsWrapper` decorates an environment to compute diagnostic telemetry on every tick:

```python
import gymnasium as gym
from netforge_rl.diagnostics.wrapper import DiagnosticsWrapper

base_env = gym.make("NetForge/Blue-v4")
env = DiagnosticsWrapper(base_env)

obs, info = env.reset(seed=42)
obs, reward, term, trunc, info = env.step(env.action_space.sample())

# Access privileged diagnostics in info
print("True compromised hosts:", info.get("oracle_compromised_count"))
print("Information asymmetry metric:", info.get("information_asymmetry"))
```

## Information Asymmetry Metric

The `information_asymmetry` score quantifies the divergence between defender beliefs and reality:

$$\text{Asymmetry} = 1 - \frac{|\text{Believed Compromises} \cap \text{True Compromises}|}{|\text{True Compromises}|}$$

- A score of `0.0` indicates perfect situational awareness (Blue has detected every compromised host).
- A higher score indicates stealthy adversary penetration that has escaped SIEM detection.

## Policy Capability Cards

The diagnostics suite also generates capability radar cards (`netforge test-policy`), scoring agents across safety, out-of-distribution robustness, false positive discipline, and adaptation.
