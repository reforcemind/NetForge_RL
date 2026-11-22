# Capability Probes

NetForge evaluates defensive agents across eight core behavioral competencies, producing normalized scores in $[0, 1]$ and diagnostic radar charts.

## Running Capability Tests

Run the capability evaluation suite on a policy:

```bash
netforge test-policy --policy heuristic-blue --out cards/
```

This generates a JSON capability card and, if `matplotlib` is installed, a radar plot saved to `cards/heuristic-blue.png`.

## Competency Dimensions

1. **Safety**: Uptime maintenance under intense attack without resorting to catastrophic outages or triggering kinetic destruction.
2. **OOD Generalization**: Performance stability when evaluating on larger, unseen topologies and randomized subnet layouts.
3. **Memory & Correlation**: Ability to correlate delayed, distributed SIEM alerts over time to detect multi-stage intrusions.
4. **False Positive Discipline**: Precision in defensive isolation, avoiding unnecessary host quarantines.
5. **Temporal Dynamics**: Effective handling of multi-tick action latencies and preemption windows.
6. **Deception Exploitation**: Efficacy in leveraging decoys, honeytokens, and misinformation to misdirect adversaries.
7. **Adaptation**: Quick recovery and strategy adjustment when the adversary shifts tactics mid-episode.
8. **Attention / Focus**: Efficacy in prioritizing high-value assets (Domain Controllers, clinical databases, PLCs) over low-impact perimeter noise.
