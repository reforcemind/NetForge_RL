# Training with Ray RLlib

NetForge provides a seamless wrapper for integrating with Ray RLlib, allowing you to train highly scalable Multi-Agent Reinforcement Learning (MARL) policies across distributed clusters.

## 1. The RLlib Environment Wrapper

NetForge implements the `MultiAgentEnv` interface via `NetForgeRLlibEnv`. This handles the conversion of NetForge's core `EnvState` into observation spaces RLlib algorithms expect (e.g., flattened dictionaries, MultiDiscrete spaces).

```python
from netforge_rl.bridges.rllib_bridge import NetForgeRLlibEnv

# The environment can be instantiated directly or via the registry
env = NetForgeRLlibEnv({
    "scenario_type": "ransomware",
    "max_ticks": 100
})
```

## 2. Registering the Environment

Before training, register the environment with Ray's global registry so worker nodes can instantiate it:

```python
import ray
from ray.tune.registry import register_env
from netforge_rl.bridges.rllib_bridge import NetForgeRLlibEnv

ray.init()

def env_creator(env_config):
    return NetForgeRLlibEnv(env_config)

register_env("netforge-v0", env_creator)
```

## 3. Configuring the Training Loop

You can train NetForge using PPO, MAPPO, or QMIX. Below is an example configuration for multi-agent PPO where the Red and Blue teams learn separate policies.

```python
from ray.rllib.algorithms.ppo import PPOConfig

# Define policy mappings
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

## 4. Recurrent Neural Networks (LSTMs)

Cybersecurity is a highly Partially Observable Markov Decision Process (POMDP). Both Red and Blue agents must maintain memory of past states. You can enable LSTMs natively in RLlib:

```python
config.training(
    model={
        "use_lstm": True,
        "max_seq_len": 20,
        "lstm_cell_size": 256,
    }
)
```

For advanced users, you can bypass Ray RLlib entirely and use the JAX-native `JaxMARL` backend for maximum throughput. See the JAX vectorization guides in the benchmarks folder.