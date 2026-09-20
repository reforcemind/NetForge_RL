# OT/SCADA physics

First-order lag on `PLC_Firmware` hosts (`netforge_rl/scenarios/ot_physics.py`).
Ticked after actions, before termination.

`x[t+1] = x[t] + (x_sp − x[t]) / τ + N(0, σ)`

| | Nominal | τ | σ | Alarm | Critical |
|---|---|---|---|---|---|
| temperature °C | 40–60 | 20 | 0.1 | > 80 | > 120 |
| pressure bar | 90–110 | 10 | 0.2 | >130 or <70 | >180 or <30 |
| flow L/min | 40–60 | 5 | 0.3 | >90 or <20 | >150 or <5 |

Alarm → SIEM severity 7. Critical → `kinetic_destruction`, severity 10, ends `ot_stuxnet`.

`OverloadPLC` (id 20, Root on PLC) jumps setpoints. ~10–15 ticks to destruction.
`IsolateHost` stops physics on that PLC.
