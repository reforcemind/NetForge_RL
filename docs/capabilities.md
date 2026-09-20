# Capability tests

```bash
netforge test-policy --policy heuristic-blue --out cards/
```

Scores in `[0, 1]`: Safety, OOD, Memory, FPs, Temporal, Deception, Adaptation,
Attention. Radar if matplotlib is installed. Full-state graphs stay on
`DiagnosticsWrapper`.
