# Threat Model & MITRE ATT&CK Alignment

NetForge RL is heavily mapped to the **MITRE ATT&CK framework**, enforcing realistic constraints on agent behavior. Exploits are not abstract probabilities; they are bound to specific CVEs and infrastructure configurations.

## MITRE ATT&CK Tactics Simulated

### 1. Initial Access & Execution
- **Spearphishing**: Targets corporate endpoints, bypassing external firewalls. Success is governed by human vulnerability scores.
- **Remote Exploits**: Simulates CVE-2019-0708 (BlueKeep) against RDP and MS17-010 (EternalBlue) against SMB.
- **Web Exploits**: Simulates HTTP Remote File Inclusions (RFI) against DMZ web servers.

### 2. Privilege Escalation
Agents frequently gain access as standard `User` and must escalate to `Root` / `SYSTEM`.
- **JuicyPotato**: Abuses DCOM on Windows (`SeImpersonatePrivilege`).
- **V4L2 Exploit**: Targets memory corruption in outdated Linux kernel modules.

### 3. Lateral Movement
- **Pass the Hash**: Once Active Directory or a Domain Controller is compromised, Red agents can extract NTLM/Kerberos hashes and move laterally without needing to trigger noisy CVE-based exploits.

### 4. Defense Evasion & Deception
Blue agents proactively counter Red movement using the MITRE Engage framework:
- **Honeytokens**: Blue injects fake, highly monitored credentials. If Red dumps `LSASS` and consumes the token, an immediate, unmaskable 100% confidence SIEM alert is generated.
- **Decoys**: Deployment of fake Apache/Tomcat/SSH daemons to sinkhole port scans and waste Red agent action economy.

## Partial Observability Constraints

Both teams suffer from severe **Partial Observability**:
- **Red Team**: Cannot see host vulnerabilities until a successful discovery action is executed.
- **Blue Team**: Only sees Red actions if they generate a SIEM log. If Red uses stealthy techniques or exploits unmonitored ports, Blue remains unaware until later stages of the attack lifecycle.