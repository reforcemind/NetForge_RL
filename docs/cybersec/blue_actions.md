# Blue Actions

Defensive actions enable Blue operators to investigate anomalies, contain active intrusions, harden configurations, and deploy deception assets.

## Action Catalog

| Action | Category | Target | Requirements | Effect |
|---|---|---|---|---|
| `IsolateHost` | Containment | Host | None | Cuts all inbound and outbound network connectivity; halts active exploits |
| `RestoreHost` | Containment | Host | Host isolated | Re-enables network connectivity, restoring operational SLA availability |
| `ConfigureACL` | Mitigation | Subnet | None | Updates subnet firewall rules to block vulnerable ports |
| `Remove` | Eradication | Host | Host compromised | Terminates adversary processes and revokes unauthorized privileges |
| `RestoreFromBackup` | Recovery | Host | None | Re-images host from known-clean golden state; resets compromise flags |
| `RotateKerberos` | Identity | Domain Controller | Domain Controller | Invalidates all active Kerberos tickets, breaking adversary lateral access |
| `SecurityAwarenessTraining` | Hardening | Subnet | None | Decreases user susceptibility to spear-phishing campaigns |
| `Analyze` | Investigation | Host | None | Inspects host processes and logs, revealing ground-truth status in info |
| `Monitor` | Investigation | Subnet | None | Increases SIEM logging verbosity and detection sensitivity on subnet |
| `DeployDecoy` | Deception | Subnet | Padding slot | Activates a decoy host (`169.254.x.x`) that alerts when touched |
| `DeployHoneytoken` | Deception | Host | Host online | Injects fake credentials into memory; alerts immediately if leveraged |
| `DecoyApache` | Deception | Host | None | Runs emulated web service sinkhole to capture exploit attempts |
| `DecoySSHD` | Deception | Host | None | Runs emulated SSH service sinkhole to capture brute-force scans |
| `DecoyTomcat` | Deception | Host | None | Runs emulated application server sinkhole |
| `Misinform` | Deception | Host | None | Returns false service banners during adversary port scans |

## Defensive Trade-offs

- **Containment vs. Availability**: `IsolateHost` instantly stops adversary lateral movement or data exfiltration from that host, but incurs a continuous SLA uptime deduction for every tick the host remains isolated.
- **False Positives**: Quarantining an uncompromised host incurs a false positive penalty and unnecessarily degrades business services.
- **Cost of Analysis**: Investigation actions take operator bandwidth without actively stopping an intrusion; defenders must balance triage against swift containment.
