# Environment Overview

NetForge RL defines a zero-sum, asymmetric, multi-agent environment with partial observability. 

## Red Policy Characteristics

- **Objective**: Maximize scalar reward by successfully executing impact actions (e.g., data exfiltration, service disruption) on designated target nodes.
- **Initial State**: Zero network visibility. `GlobalNetworkState` is fully masked.
- **Mechanics**: Agents must sequentially execute discovery, exploit, and privilege escalation actions to modify node states. Success probabilities are calculated deterministically against the target's vulnerability array and current access level.

## Blue Policy Characteristics

- **Objective**: Maximize scalar reward by maintaining service uptime and minimizing compromised host counts across the episode.
- **Initial State**: Full visibility of the uncompromised network baseline.
- **Mechanics**: Agents observe the environment strictly through SIEM event logs. Actions consist of isolation, firewall ACL modification, and host restoration. Detection capabilities are dependent on Red action signatures; stealthy actions may bypass event log generation.

## Kinetic Impacts

The environment supports Cyber-Physical System (CPS) nodes. Specific Red actions (`OverloadPLC`) can transition CPS nodes into a `kinetic_destruction` state. This represents a terminal fail-state for Blue agents, immediately ending the episode.