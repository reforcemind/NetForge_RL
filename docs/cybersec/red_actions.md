# Red Actions

Adversary actions model the phases of the cyber kill chain, from initial reconnaissance to objective impact.

## Action Catalog

| Action | Category | Target | Requirements | Effect |
|---|---|---|---|---|
| `DiscoverRemoteSystems` | Reconnaissance | Subnet | None | Discovers responsive IP addresses within a target subnet |
| `DiscoverNetworkServices` | Reconnaissance | Host | Host discovered | Enumerates listening ports and running service banners |
| `NetworkScan` | Reconnaissance | Subnet | None | Wide-spectrum scan revealing reachable topology |
| `ExploitRemoteService` | Exploitation | Host | Host & port open | Exploits generic vulnerability to gain `User` access |
| `ExploitBlueKeep` | Exploitation | Host | Port 3389 (RDP) & CVE | Exploits RDP flaw for direct privilege elevation |
| `ExploitEternalBlue` | Exploitation | Host | Port 445 (SMB) & CVE | Exploits SMBv1 flaw for direct `Root` access |
| `HTTP_RFI` | Exploitation | Host | Port 80/443 & CVE | Remote file inclusion gaining initial web shell |
| `PrivilegeEscalate` | Escalation | Host | `User` privilege | Attempts local privilege escalation to `Root` |
| `JuicyPotato` | Escalation | Host | `User` & Windows DCOM | Exploits DCOM subsystem to impersonate `SYSTEM` |
| `V4L2KernelExploit` | Escalation | Host | `User` & Linux kernel | Exploits kernel driver flaw to achieve `Root` |
| `DumpLSASS` | Post-Exploit | Host | `Root` privilege | Dumps cached Kerberos/NTLM authentication tokens |
| `PassTheHash` | Lateral Movement | Host | Valid NTLM hash | Authenticates to remote host using captured hash |
| `PassTheTicket` | Lateral Movement | Host | Valid Kerberos ticket | Authenticates to remote host using captured ticket |
| `ShareIntelligence` | Coordination | Agents | Red agent | Merges discovered topology visibility between Red agents |
| `ExfiltrateData` | Impact | Host | `User` or `Root` | Exfiltrates confidential data chunks, earning Red reward |
| `Impact` | Impact | Host | `Root` privilege | Encrypts local files or shuts down host services |
| `KillProcess` | Impact | Host | `Root` privilege | Terminates active security agents or logging daemons |
| `OverloadPLC` | Kinetic | Host | `Root` on PLC host | Modifies sensor setpoints to induce physical overpressure |
| `SpearPhishing` | Social Eng. | Host | None | Targets human vulnerability score to gain initial foothold |

## Durative Mechanics

Red actions require execution ticks to resolve:

1. **Submission**: The adversary submits an action `[action_type, target_host_index]`.
2. **Queueing**: The action enters the pending queue with a defined duration (e.g., `DiscoverNetworkServices` takes 2 ticks; `ExfiltrateData` takes 3 ticks).
3. **Maturity & Preemption**: If the host is isolated by Blue before the action completes, the pending action is cancelled. If it matures without preemption, state changes are committed to the network.
