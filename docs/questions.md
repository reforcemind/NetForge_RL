# Research Probes

NetForge includes experimental probes in `netforge_rl/arena/questions/` designed to evaluate specific reinforcement learning hypotheses in cybersecurity.

```{image} _static/figures/belief.svg
:alt: SIEM belief state versus full oracle state
:class: fe-fig
```

## Running Probes

List available research questions or run an experiment via the CLI:

```bash
# List all active research probes
netforge questions

# Run the belief-state vs oracle experiment
netforge questions belief-vs-oracle --seeds 0 --max-ticks 40
```

## Available Probe Experiments

| Probe ID | Research Hypothesis |
|---|---|
| `belief-vs-oracle` | Quantifies the observation gap: how much compromised network state is invisible to Blue under pure SIEM telemetry |
| `reward-design` | Compares policy stability under pure SLA reward vs. false-positive penalized reward formulations |
| `red-population` | Evaluates defense robustness when facing an adversarial population rather than a single fixed script |
| `risk-sensitive` | Analyzes policy risk profiles by comparing expected return against Conditional Value at Risk (CVaR) |
| `time-mode` | Compares fixed-step tick updates against event-driven time jumps |
| `ood-topology` | Tests policy generalization when deployed onto network topologies larger than seen during training |
| `adaptation` | Evaluates defender response when Red dynamically shifts attack strategies mid-episode |
| `deception` | Measures defender resilience and adversary delay when honeypots and decoy hosts are deployed |
| `constrained-sla` | Formulates defense as constrained optimization, maximizing security under hard SLA availability bounds |
| `gnn-belief` | Evaluates Graph Neural Network policies trained on reconstructed belief graphs |
| `offline-to-online` | Pretrains policies on static offline trajectory dumps followed by online fine-tuning |
| `league-psro` | Computes Policy-Space Response Oracles (PSRO) across leagues of Red and Blue policies |
