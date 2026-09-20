# Probes

Small experiments the gym already runs. Heuristics are in-tree so they execute.

```{image} _static/figures/belief.svg
:alt: SIEM belief vs full state
:class: fe-fig
```

```bash
netforge questions
netforge questions belief-vs-oracle --seeds 0 --max-ticks 40
```

| id | What it checks |
|---|---|
| `belief-vs-oracle` | How much compromise is missing from Blue’s SIEM graph |
| `reward-design` | SLA-only / FP-penalized reward vs isolation |
| `red-population` | One Blue vs several Reds |
| `risk-sensitive` | Mean vs CVaR vs kinetic tails |
| `time-mode` | Event-driven vs fixed-step |
| `ood-topology` | More live hosts than training |
| `adaptation` | Red changes strategy mid-episode |
| `deception` | Honeypots / decoys |
| `constrained-sla` | Defense under SLA / FP costs |
| `gnn-belief` | Train a GNN on SIEM belief graphs (your trainer) |
| `offline-to-online` | Pretrain on dumps, then online |
| `league-psro` | Best response to a Red league |

Add one in `netforge_rl/arena/questions/`. {doc}`submissions`
