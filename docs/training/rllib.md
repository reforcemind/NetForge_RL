# Ray RLlib

```python
from netforge_rl.bridges.rllib_bridge import NetForgeRLlibEnv

env = NetForgeRLlibEnv({"scenario_type": "ransomware", "max_ticks": 100})
```

Reference: `python benchmarks/rllib_rmappo.py` — shared LSTM PPO, red + blue,
`train_batch_size=1024`, legacy API stack.

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
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
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
```

Curriculum: {doc}`curriculum`.
