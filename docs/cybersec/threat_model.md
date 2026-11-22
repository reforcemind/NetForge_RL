# Threat Model

The NetForge threat model defines the operational rules, information boundaries, and physical mechanics governing interactions between agents and the network.

## Network Topology & Zoning

Networks are organized into structured security zones connected via routing and firewall policy:

- **Perimeter / DMZ**: Hosts directly exposed to inbound external traffic (web servers, reverse proxies).
- **Corporate LAN**: User workstations, local file shares, and print servers.
- **Restricted Zone**: Domain Controllers, customer databases, and proprietary intellectual property.
- **OT / Industrial Zone**: Engineering workstations, Human-Machine Interfaces (HMIs), and PLCs.

Subnet boundaries restrict direct traffic flow. An adversary cannot directly interact with Restricted or OT hosts from the outside without establishing intermediary pivot points within DMZ or Corporate hosts.

## Host State & Privileges

Each host in the simulation maintains explicit state tracking:

| Attribute | Possible Values | Description |
|---|---|---|
| `compromised_by` | `'None'`, `'red_operator'` | Ownership flag for active compromise |
| `privilege` | `'None'`, `'User'`, `'Root'` | Privilege tier on the host |
| `status` | `'online'`, `'isolated'`, `'offline'` | Network connectivity status |
| `os_type` | `'Linux'`, `'Windows'`, `'PLC_Firmware'` | Operating system platform |
| `cve_profile` | Bitmask of active CVEs | Known exploitable vulnerabilities present |
| `tokens` | Array of session credentials | Active Kerberos / NTLM tokens cached on host |

## Identity & Credential Movement

NetForge enforces an explicit credential lifecycle:

- Workstations and servers store authentication tokens upon user login.
- Adversaries with `Root` privileges can extract tokens via `DumpLSASS`.
- Harvested tokens enable credential-based lateral movement (`PassTheHash`, `PassTheTicket`) bypassing service vulnerability requirements.
- Defenders can counter identity theft by rotating master ticket-granting keys with `RotateKerberos`, invalidating all active adversary tokens.

## Telemetry & SIEM Generation

Actions generate observable events based on telemetry rules:

- **Authentication Events (Event ID 4624 / 4625)**: Generated during remote login, credential abuse, or brute-force attempts.
- **Process Creation (Event ID 4688 / Sysmon 1)**: Emitted during privilege escalation and tool execution.
- **Network Traffic (Sysmon 3)**: Logged during port scanning, service discovery, and data exfiltration.
- **Configurable Noise & Latency**: Scenarios introduce benign background noise and delayed log ingestion, requiring defenders to correlate events across time rather than relying on instantaneous alerts.

## Deception Architecture

Defenders can deploy active deception mechanisms:

- **Decoy Nodes**: Simulated dummy servers occupying padding IP space (`169.254.0.0/16`). Scans targeting decoys immediately flag adversary presence.
- **Honeytokens**: Bogus credentials placed in memory. When accessed or leveraged by an adversary, they generate unmaskable high-severity alerts.
