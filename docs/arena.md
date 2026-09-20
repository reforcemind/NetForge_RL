# Evaluation splits

Train, check, and hold out topologies with the same metrics: mission, SLA,
false positives, exfil, kinetic failures.

```{image} _static/figures/arena.svg
:alt: train / dev / hidden splits
:class: fe-fig
```

```bash
netforge arena --json
netforge evaluate examples/submissions/heuristic_blue.py --seeds 0 1 --max-ticks 40
netforge benchmark --table --seeds 0 1 --max-ticks 40
```

| Split | Public | Use |
|---|---|---|
| `train` | yes | learning |
| `dev` | yes | pick a checkpoint |
| `hidden` | organizer | official eval (repo ships a public proxy) |

Blue is scored on keeping the network up without wrecking it (isolation / FPs).
Red eval can use a **population** (`random`, heuristic, kill-chain), not one
script. {doc}`submissions` · {doc}`questions`
