# Cybersecurity Overview

NetForge RL simulates realistic enterprise networks where autonomous AI agents compete in a dynamic, zero-sum environment. The environment is asymmetric, capturing the true "Fog of War" experienced during cyber operations.

## The Red Team

The **Red Team** acts as the Advanced Persistent Threat (APT). Their objective is to compromise hosts, escalate privileges, and execute a final impact objective (such as deploying ransomware or exfiltrating data). 

- **Starting State**: Red agents begin with no initial network access. They must discover the perimeter via scanning or spearphishing.
- **Progression**: They operate via the Cyber Kill Chain: Reconnaissance -> Initial Access -> Privilege Escalation -> Lateral Movement -> Impact.
- **Visibility**: Red agents only see what they explicitly discover through active scanning (`NetworkScan`, `DiscoverRemoteSystems`) or what is passed to them via allied intelligence sharing.

## The Blue Team

The **Blue Team** acts as the Security Operations Center (SOC). Their objective is to maintain network uptime, preserve system integrity, and eliminate Red team presence.

- **Starting State**: Blue agents possess a baseline understanding of their assigned subnets.
- **Progression**: They monitor SIEM (Security Information and Event Management) logs. When anomalous behavior is detected, they can isolate compromised hosts (`IsolateHost`), block traffic (`ConfigureACL`), deploy honeypots (`DeployDecoy`), or restore machines (`RestoreFromBackup`).
- **Visibility**: Blue agents rely heavily on the SIEM. If a Red agent bypasses endpoint detection, the Blue agent remains blind to the intrusion until an observable action generates a log.

## Kinetic Impacts (OT/ICS)

Unlike traditional IT networks, NetForge simulates Cyber-Physical Systems (CPS). Red agents can compromise SCADA frameworks to cause physical kinetic destruction (e.g., `OverloadPLC`). If a PLC is physically destroyed, it is a terminal fail-state for the Blue Team.