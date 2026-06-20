# Training with Ray RLlib

## Environment Wrapper

NetForge implements the RLlib `MultiAgentEnv` interface via `NetForgeRLlibEnv`. This wrapper automatically maps the `EnvState` PyTree into RLlib-compatible observation spaces (flattened dictionaries, MultiDiscrete spaces).

```python
from netforge_rl.bridges.rllib_bridge import NetForgeRLlibEnv

env = NetForgeRLlibEnv({
    "scenario_type": "ransomware",
    "max_ticks": 100
})
```

## Global Registry

Register the environment for distributed worker nodes:

```python
import ray
from ray.tune.registry import register_env
from netforge_rl.bridges.rllib_bridge import NetForgeRLlibEnv

ray.init()

def env_creator(env_config):
    return NetForgeRLlibEnv(env_config)

register_env("netforge-v0", env_creator)
```

## PPO Training Configuration

Example configuration for Multi-Agent PPO mapping independent policies to Red and Blue agents.

```python
from ray.rllib.algorithms.ppo import PPOConfig

def policy_mapping_fn(agent_id, *args, **kwargs):
    if "red" in agent_id:
        return "red_policy"
    else:
        return "blue_policy"

config = (
    PPOConfig()
    .environment("netforge-v0", env_config={"scenario_type": "ransomware"})
    .multi_agent(
        policies={"red_policy", "blue_policy"},
        policy_mapping_fn=policy_mapping_fn,
    )
    .resources(num_gpus=1)
    .rollouts(num_rollout_workers=4)
)

algo = config.build()

for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: Red Reward: {result['policy_reward_mean']['red_policy']:.2f}")
```

## RNN Integration

POMDP characteristics require memory. Enable LSTMs within the RLlib model configuration:

```python
config.training(
    model={
        "use_lstm": True,
        "max_seq_len": 20,
        "lstm_cell_size": 256,
    }
)
```