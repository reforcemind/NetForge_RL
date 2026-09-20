# Red actions

| Action | Module | |
|---|---|---|
| `DiscoverRemoteSystems` / `NetworkScan` | reconnaissance | subnet IPs → visibility |
| `DiscoverNetworkServices` | reconnaissance | services on a host |
| `ExploitBlueKeep` / `EternalBlue` / `HTTP_RFI` | exploits | RDP / SMB / HTTP vs CVE flags |
| `ExploitRemoteService` | exploits | network vis → User |
| `PrivilegeEscalate` | privilege_escalation | User → Root |
| `JuicyPotato` / `V4L2KernelExploit` | privilege_escalation | Windows DCOM / Linux kernel |
| `DumpLSASS` | post_exploitation | tokens; needs Root |
| `PassTheHash` / `PassTheTicket` | post_exploitation | tokens, not CVEs |
| `ShareIntelligence` | coordination | union of visibility masks |
| `ExfiltrateData` / `Impact` / `KillProcess` | impact | terminal objectives |
| `OverloadPLC` | kinetic | PLC → `kinetic_destruction` |
| `SpearPhishing` | social_engineering | vs `human_vulnerability_score`; no routing |
