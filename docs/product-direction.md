# Scope & Philosophy

NetForge is focused on solving a specific, critical bottleneck in cybersecurity reinforcement learning: providing defenders with realistic information constraints and operational metrics.

## Core Principles

1. **Defenders Observe Telemetry, Not Oracles**: In real-world security operations, defenders do not possess global knowledge of attacker positions. Training defenders on raw compromise maps teaches unrealistic policies that fail when deployed against real telemetry.
2. **Defenders Must Respect Availability**: A defender that "wins" by immediately disconnecting every server on the network destroys business operations. Availability (SLA) and false-positive rates must constrain containment actions.
3. **Actions Take Time**: Network scans, remote exploits, and defensive patching operate across time. Durative actions with preemption windows reflect real operational timing.
4. **Clean Boundaries**: NetForge provides the environment, benchmarks, and evaluation contracts. Algorithms (PPO, QMIX, GNNs) and proprietary threat data remain external.
