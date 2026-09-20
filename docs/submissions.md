# Your agent

Drop in a Blue policy with `make_blue`. Eval reports mission, SLA, FPs, security,
kinetic failures — not only return.

```bash
netforge evaluate examples/submissions/heuristic_blue.py --seeds 0 1 --max-ticks 40
```

```python
NAME, TEAM, TECHNIQUE = 'my-blue', 'lab', 'ppo-belief'


def wrap(env):
    return env  # belief graphs already on; do not attach oracle_graph


def make_blue(seed=0):
    return MyPolicy(seed=seed)
```

`act(env, agent_id) -> np.ndarray`. Optional `make_red`. `--card` adds probes.

{doc}`arena` · `examples/submissions/heuristic_blue.py`
