# Quick Start Guide

## Installation

Requires Python 3.12+.

```bash
pip install 'netforge_rl[jax,render,finetune] @ git+https://github.com/reforcemind/NetForge_RL'
```

Extras: `jax` (vectorized backend), `render` (matplotlib + moviepy), `finetune`
(TRL/PEFT for LLM agents), `rllib` (Ray RLlib bridge).

## PettingZoo execution (single environment)

The reference Python backend implements the PettingZoo parallel API. Each agent acts with a
`MultiDiscrete([32, 100])` action — `[action_type_id, target_host_index]`.

```python
import numpy as np
from netforge_rl.environment.parallel_env import NetForgeRLEnv

env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 50})
obs, infos = env.reset(seed=0)

while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, term, trunc, infos = env.step(actions)
    if all(term.values()) or all(trunc.values()):
        break
```

Each `obs[agent]` is a dict with `obs` (256-d state), `action_mask` (132-d), `siem_embedding`
(128-d), `adj_matrix`, and `delta_t`; Blue agents additionally get `blue_comm`.

## Difficulty presets and the held-out split

Instead of hand-writing config dicts, use named difficulty tiers and the frozen evaluation
seed suite. See [Difficulty & Splits](environment/difficulty.md) for the full knob table.

```python
from netforge_rl.environment import make_env, EVAL_SEEDS

# Train on any seeds at a chosen difficulty
train_env = make_env('medium', scenario_type='apt_espionage', seed=0)

# Report on held-out topologies never seen during training
eval_env = make_env('hard', scenario_type='apt_espionage', evaluation=True, seed=EVAL_SEEDS[0])
```

## JAX vectorized execution

Hardware-accelerated batched rollouts for high-throughput training.

```python
import jax
from netforge_rl.backends.jax import VectorEnvSpec
from netforge_rl.bridges.jaxmarl import JaxMARLEnv, random_action_dict

env = JaxMARLEnv(spec=VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3), batch_size=4096)
key = jax.random.PRNGKey(0)

obs, state = env.reset(key)
obs, state, reward, done, info = env.step(key, state, random_action_dict(env, key))
```

## Running baselines

```python
from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.baselines.policies import KillChainRedPolicy, HeuristicBluePolicy

env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 150})
env.reset(seed=0)
red, blue = KillChainRedPolicy(seed=0), HeuristicBluePolicy(seed=0)

while env.agents:
    actions = {a: (red.act(env, a) if 'red' in a else blue.act(env, a)) for a in env.agents}
    _, _, term, trunc, _ = env.step(actions)
    if all(term.values()) or all(trunc.values()):
        break
```

`KillChainRedPolicy` runs a recon → exploit → pivot kill-chain and actually compromises
hosts; see [Baselines](benchmarks/baselines.md).

## Single-agent training (Gymnasium / SB3 / CleanRL)

For standard single-agent RL, `NetForgeSingleAgentEnv` is a `gymnasium.Env` that controls one
agent against scripted opponents. It passes `gymnasium`'s `check_env` and exposes the action
mask in `info` for maskable algorithms.

```python
from netforge_rl.environment import NetForgeSingleAgentEnv
from netforge_rl.baselines.policies import KillChainRedPolicy

env = NetForgeSingleAgentEnv(
    'ransomware',
    controlled_agent='blue_dmz',
    opponents={'red_operator': KillChainRedPolicy(seed=0)},
)
obs, info = env.reset(seed=0)          # obs: Box(384,), info['action_mask']: (132,)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Training IPPO on the JAX backend

```python
from netforge_rl.baselines.jax_ppo import ippo_train, PPOConfig, save_params

out = ippo_train(PPOConfig(total_iters=40, num_steps=48, batch_size=128))
save_params(out['params'], 'runs/ippo_blue.npz')
print('reward:', out['reward_curve'][0], '->', out['reward_curve'][-1])
```

See [Baselines](benchmarks/baselines.md) for the committed learning curve.

## Running the diagnostic probes

```python
from netforge_rl.diagnostics import all_diagnostics, run_diagnostic
from netforge_rl.baselines.policies import HeuristicBluePolicy

for probe in all_diagnostics():
    result = run_diagnostic(probe, HeuristicBluePolicy(seed=0), seed=0)
    print(f'{probe.capability:14} {probe.name:26} score={result.score:.2f}')
```

See [Diagnostics](diagnostics/overview.md) for what each capability measures.

## LLM SOC agent (optional)

`run_episode` takes a `clients` dict mapping each agent id to an LLM client; uncontrolled
agents no-op, and unparseable replies are counted as `invalid_replies`.

```python
from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.semantic.clients.mock import MockLLMClient
from netforge_rl.semantic.runner import run_episode

env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 50})
result = run_episode(env, clients={'blue_dmz': MockLLMClient()}, seed=0)
print(result.rewards, result.invalid_replies)
```

Swap `MockLLMClient` for an OpenAI/Anthropic/vLLM client to have a language model read raw
SIEM logs and issue defensive actions.
