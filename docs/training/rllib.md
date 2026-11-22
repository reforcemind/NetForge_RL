# Ray RLlib Integration

NetForge includes an environment bridge for [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) in `netforge_rl.bridges.rllib_bridge.NetForgeRLlibEnv`.

## Training Example

The following script registers NetForge with Ray Tune and trains multi-agent PPO policies:

```python
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from netforge_rl.bridges.rllib_bridge import NetForgeRLlibEnv

ray.init()

# Register the environment factory
register_env("netforge-v0", lambda cfg: NetForgeRLlibEnv(cfg))

# Configure multi-agent PPO
config = (
    PPOConfig()
    .environment("netforge-v0", env_config={"scenario_type": "ransomware"})
    .multi_agent(
        policies={
            "red_policy": (None, None, None, {"model": {"use_lstm": True}}),
            "blue_policy": (None, None, None, {"model": {"use_lstm": True}}),
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: (
            "red_policy" if "red" in agent_id else "blue_policy"
        ),
    )
    .training(train_batch_size=1024)
)

algo = config.build()

for iteration in range(50):
    result = algo.train()
    print(f"Iter {iteration}: mean reward = {result.get('env_runners', {}).get('episode_reward_mean', 0):.2f}")
```

For a complete reference implementation using shared LSTM policies, see `benchmarks/rllib_rmappo.py`.
