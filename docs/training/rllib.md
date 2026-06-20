# Training with Ray RLlib

## Environment Wrapper

NetForge implements the RLlib `MultiAgentEnv` interface via `NetForgeRLlibEnv`.

```python
from netforge_rl.bridges.rllib_bridge import NetForgeRLlibEnv

env = NetForgeRLlibEnv({"scenario_type": "ransomware", "max_ticks": 100})
```

## RMAPPO Baseline

The reference training script is [`benchmarks/rllib_rmappo.py`](../../benchmarks/rllib_rmappo.py). It trains two shared LSTM policies (one red, one blue) with PPO on the legacy API stack.

```bash
python benchmarks/rllib_rmappo.py
```

Key settings: `train_batch_size=1024`, `use_lstm=True`, legacy API stack (`enable_rl_module_and_learner=False`). Custom metrics from `NetForgeMetricsCallback` are logged per-agent each step.

## PPO Configuration

```python
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from netforge_rl.bridges.rllib_bridge import NetForgeRLlibEnv

ray.init()
register_env("netforge-v0", lambda cfg: NetForgeRLlibEnv(cfg))

config = (
    PPOConfig()
    .environment("netforge-v0", env_config={"scenario_type": "ransomware"})
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .multi_agent(
        policies={
            "red_rmappo": (None, None, None, {"model": {"use_lstm": True}}),
            "blue_rmappo": (None, None, None, {"model": {"use_lstm": True}}),
        },
        policy_mapping_fn=lambda agent_id, *a, **kw: (
            "red_rmappo" if "red" in agent_id else "blue_rmappo"
        ),
    )
    .training(train_batch_size=1024)
)
algo = config.build()
result = algo.train()
```

## Curriculum Learning

`CurriculumWrapper` wraps `NetForgeRLEnv` and automatically progresses through three phases based on rolling mean episode reward.

| Phase | Active hosts | Scenarios | DHCP | Dynamic topology | Reward scale | Graduate at |
|---|---|---|---|---|---|---|
| `novice` | 5 | ransomware | off | off | 3× | mean ≥ 60 over 10 eps |
| `intermediate` | 25 | ransomware, apt | 80 ticks | off | 1.5× | mean ≥ 40 over 15 eps |
| `expert` | 100 | ransomware, apt, iot | 40 ticks | churn + migration | 1× | final |

```python
from netforge_rl.environment.curriculum import CurriculumWrapper

env = CurriculumWrapper(
    base_cfg={"max_ticks": 200},
    on_phase_advance=lambda idx, name: print(f"Phase {idx}: {name}"),
)
obs, _ = env.reset(seed=0)
```

Each step's `info` dict contains `__curriculum__` with `phase`, `phase_index`, `mean_reward`, `window_fill`, and `phase_advanced`.

Start from a specific phase or supply custom phases:

```python
from netforge_rl.environment.curriculum import CurriculumWrapper, PHASES

env = CurriculumWrapper(phases=PHASES, start_phase=1)  # start at intermediate
```
```