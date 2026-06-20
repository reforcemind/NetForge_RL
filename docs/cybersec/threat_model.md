# Action Taxonomy Mapping

NetForge RL actions are mapped to specific execution mechanics rather than abstract probabilities.

## Implemented Vectors

### Initial Access & Execution
- **Spearphishing**: Evaluated against the `human_vulnerability_score` scalar of the target end-user node. Bypasses routing constraints.
- **Remote Exploits**: 
  - `ExploitBlueKeep` (CVE-2019-0708): Modifies access state on RDP.
  - `ExploitEternalBlue` (MS17-010): Modifies access state on SMB.
  - `ExploitHTTP_RFI`: Modifies access state on HTTP/HTTPS.

### Privilege Escalation
Agents transitioning from `User` to `Root`/`SYSTEM` state requirements:
- **JuicyPotato**: Exploits DCOM (`SeImpersonatePrivilege`) on Windows nodes.
- **V4L2 Kernel Exploit**: Targets specific kernel array flags on Linux nodes.

### Lateral Movement
- **Pass the Hash**: Validates token exchanges against Domain Controller state vectors, allowing state modification without exploiting service CVEs.

### Defense Evasion & Deception
Blue agent countermeasures:
- **DeployHoneytoken**: Injects a monitored token state into the target host. Ingestion by a Red agent generates an unmaskable SIEM event alert.
- **Decoys**: Instantiates isolated nodes mimicking Apache/Tomcat/SSH services to intercept and sinkhole Red discovery actions.

## Visibility Masks
- **Red Policy**: Local knowledge graph updated only via successful discovery actions (`NetworkScan`, `DiscoverRemoteSystems`).
- **Blue Policy**: Graph updated only when an executed action triggers a SIEM log event according to its predefined signature.