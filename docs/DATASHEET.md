# NetForge Environment Datasheet

This datasheet specifies the formal parameters, observation spaces, action spaces, and evaluation dimensions of the NetForge reinforcement learning environment.

## Environment Specifications

| Parameter | Specification | Details |
|---|---|---|
| **Host Capacity** | 100 host slots | Typically 15–35 live hosts; remainder reserved for dynamic arrival / decoys (`169.254.0.0/16`) |
| **State Representation** | `GlobalNetworkState` | Full host state array, routing topology, credential tables, and continuous physical variables |
| **Scenario Families** | 5 families | `ransomware`, `apt_espionage`, `cloud_hybrid`, `iot_grid`, `ot_stuxnet` |
| **Agent Roles** | Up to 4 agents | `red_operator`, `blue_dmz`, `blue_internal`, `blue_restricted` |
| **Action Space** | `MultiDiscrete([32, 100])` | Action Code (0–31) $\times$ Target Host (0–99) |
| **Action Mask** | `int8[132]` | Valid legal actions (32 action codes + 100 host targets) |
| **Observation Dimensions** | Multiple channels | Vector: `(256,)`; SIEM Embedding: `(128,)`; Adjacency: `(10000,)` |
| **Graph Observations** | Belief Graph Dict | Node features `(100, 8)`, Edge index `(2, E)`, Edge attributes `(E, 4)` |
| **Difficulty Levels** | `easy`, `medium`, `hard` | Configures network scale, adversary stealth, telemetry latency, and background noise |
| **Evaluation Splits** | `train`, `dev`, `hidden` | Frozen seed allocations for unbiased policy evaluation |
| **Key Performance Metrics** | Composite scorecard | SLA Uptime %, Mean Time to Containment (MTTC), False Positive Rate, Exfiltration (MB) |

## Limitations & Boundaries

- **Simulated Telemetry**: NetForge generates synthetic Windows Security and Sysmon event records based on action executions; it is not a kernel hypervisor.
- **Abstract Payloads**: Exploits and malware behaviors are represented by state transition mechanics rather than live weaponized binary execution.
