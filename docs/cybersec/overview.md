# Cyber Defense Overview

NetForge formulates network defense as an Asymmetric Partially Observable Markov Decision Process (POMDP) played between an attacking adversary (Red) and one or more defensive security operators (Blue).

```
                  ┌──────────────────────┐
                  │ Global Network State │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   ┌─────────────────┐               ┌─────────────────┐
   │  Adversary Red  │               │   Defender Blue │
   │  (Fog of War)   │               │ (SIEM Telemetry)│
   └────────┬────────┘               └────────┬────────┘
            │                                 │
     Attacking Actions                 Defensive Actions
 (Scan, Exploit, Pivot)            (Isolate, ACL, Restore)
```

## Adversary Model (Red)

The adversary aims to compromise high-value assets, exfiltrate sensitive data, or disrupt critical services while minimizing detection:

- **Visibility Under Fog of War**: Red begins with zero knowledge of internal network structure, active IP addresses, or running services. It must actively probe subnets to discover reachable hosts.
- **Deterministic Exploits**: Exploits succeed only when matching prerequisites are satisfied (e.g., target service open, relevant CVE unpatched, host reachable through network routing).
- **Lateral Movement & Escalation**: Once initial access is gained with User privileges, Red must execute privilege escalation (e.g., kernel exploits, DCOM abuse) or harvest identity tokens (`DumpLSASS`) to pivot across hosts.

## Defender Model (Blue)

Defenders represent Security Operations Center (SOC) personnel protecting enterprise infrastructure:

- **Tiered Multi-Agent Defense**: In multi-agent scenarios, defense is partitioned into three specialized agents:
  - `blue_dmz`: Monitors and protects public-facing subnets, web applications, and perimeter proxies.
  - `blue_internal`: Oversees corporate workstations, internal file servers, and directory services.
  - `blue_restricted`: Guards high-security segments, database clusters, and operational technology (OT).
- **Telemetry-Driven Observation**: Blue does not observe the true network compromise map. Instead, Blue receives an aggregated SIEM buffer containing Sysmon-like events (process creations, network connections, authentication failures) subject to configurable logging latency and drop rates.
- **Operational Availability (SLA)**: Every defensive action carries operational consequences. Severing network connectivity with `IsolateHost` immediately halts an active exploit, but renders the host unavailable to legitimate business users, degrading overall SLA uptime.

## Kinetic & Industrial Domain

In operational technology scenarios (e.g., `ot_plant`, `ot_stuxnet`), the network interfaces with simulated physical processes (pressure, temperature, flow rate). Adversaries targeting programmable logic controllers (PLCs) with actions like `OverloadPLC` induce mechanical stress. Blue defenders must recognize anomalies from telemetry and isolate compromised controllers before physical damage thresholds are exceeded.
