# OT / SCADA Physics

In operational technology environments, cyber attacks frequently target physical systems controlled by Programmable Logic Controllers (PLCs). NetForge includes a continuous physics simulation layer coupled to host states in `netforge_rl/scenarios/ot_physics.py`.

## Physical State Model

Each PLC host simulates continuous process variables governed by first-order lag dynamics with stochastic Gaussian noise:

$$x_{t+1} = x_t + \frac{x_{\text{sp}} - x_t}{\tau} + \mathcal{N}(0, \sigma^2)$$

Where:
- $x_t$ is the current process value.
- $x_{\text{sp}}$ is the operational setpoint.
- $\tau$ is the system time constant (inertia).
- $\sigma$ is process measurement noise.

## Process Parameters

| Variable | Nominal Operating Range | Time Constant ($\tau$) | Alarm Threshold | Critical Failure Threshold |
|---|---|---|---|---|
| **Temperature** | 40°C – 60°C | 20 ticks | > 80°C | > 120°C |
| **Pressure** | 90 bar – 110 bar | 10 ticks | < 70 bar or > 130 bar | < 30 bar or > 180 bar |
| **Flow Rate** | 40 L/min – 60 L/min | 5 ticks | < 20 L/min or > 90 L/min | < 5 L/min or > 150 L/min |

## Attack & Defense Dynamics

- **Adversary Manipulation (`OverloadPLC`)**: When an adversary achieves `Root` privilege on a PLC host, it can execute `OverloadPLC`, forcing setpoints to catastrophic levels.
- **Alarm Progression**: When values exceed the alarm threshold, the SIEM emits high-severity telemetry events (`SCADA_PHYSICAL_ALARM`, severity 7).
- **Kinetic Destruction**: If an uncontained controller runs at critical thresholds for approximately 10 to 15 ticks, the physical plant suffers irreversible damage (`kinetic_destruction`, severity 10), triggering immediate termination and heavy Blue penalties.
- **Defensive Containment**: Issuing `IsolateHost` on the compromised PLC severs adversary control and reverts setpoints to safe fallback defaults.
