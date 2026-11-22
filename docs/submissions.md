# Policy Submissions

NetForge provides a submission contract allowing external policies (heuristics, RL models, GNNs, or LLMs) to be evaluated against standardized scenarios and adversarial populations.

## Submission Contract

To package a policy for evaluation, create a Python file exposing the metadata and factory functions:

```python
"""Submission policy module."""
import numpy as np

NAME = "my-blue-agent"
TEAM = "blue"
TECHNIQUE = "ppo-belief"


def wrap(env):
    """Optional environment wrapper.

    Belief graph observations are enabled by default.
    Do not attach privileged oracle wrappers during official evaluation.
    """
    return env


def make_blue(seed=0):
    """Instantiate and return the defensive policy."""
    return MyDefensivePolicy(seed=seed)


class MyDefensivePolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, agent_id: str) -> np.ndarray:
        """Select an action for the specified agent.

        Returns:
            np.ndarray: [action_type, target_host_index]
        """
        mask = env.infos[agent_id].get("action_mask")
        return env.action_space(agent_id).sample(mask=mask)
```

## Evaluating Submissions

Run evaluation on the `dev` split:

```bash
netforge evaluate examples/submissions/heuristic_blue.py --seeds 0 1 --max-ticks 40
```

To include capability diagnostics and generate policy cards:

```bash
netforge evaluate examples/submissions/heuristic_blue.py --card --seeds 0 1 --max-ticks 40
```
