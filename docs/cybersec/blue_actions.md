# Blue actions

| Action | Module | |
|---|---|---|
| `ConfigureACL` | mitigation | drop ports on a subnet |
| `IsolateHost` | mitigation | cut all traffic |
| `RestoreHost` | mitigation | un-isolate |
| `Remove` | mitigation | clear compromise |
| `RestoreFromBackup` | mitigation | reset to baseline |
| `SecurityAwarenessTraining` | mitigation | lower `human_vulnerability_score` |
| `DecoyApache` / `SSHD` / `Tomcat` | deception | sinkhole discovery |
| `DeployDecoy` / `DeployHoneytoken` | deception | decoy node / token → SIEM on use |
| `Misinform` | deception | lie on Red discovery |
| `RotateKerberos` | identity | wipe DC tokens |
| `Analyze` | analysis | unmasked host (diagnostic) |
| `Monitor` | analysis | raise SIEM rate on a subnet |
