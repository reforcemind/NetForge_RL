# NetForge Actions Reference

This document details all implemented actions for the Red and Blue teams within the NetForge environment.

## Blue Team Actions

### `ConfigureACL`
- **Module:** `mitigation.py`
- **Description:** Dynamically modifies the implicit routing Firewall to block specific port traffic inbound to a protected subnet.

### `IsolateHost`
- **Module:** `mitigation.py`
- **Description:** Disconnects a compromised host completely from the network infrastructure.

### `Remove`
- **Module:** `mitigation.py`
- **Description:** Evicts unauthorized threat actors from a compromised element.

### `RestoreFromBackup`
- **Module:** `mitigation.py`
- **Description:** Executes a bare-metal imaging recovery to purge advanced persistent threats (APTs).

### `RestoreHost`
- **Module:** `mitigation.py`
- **Description:** Re-establishes network connectivity for a previously isolated host.

### `SecurityAwarenessTraining`
- **Module:** `mitigation.py`
- **Description:** Deploys rapid, intensive anti-phishing training to a targeted subnet.

## Blue_commander Team Actions

### `DecoyApache`
- **Module:** `deception.py`
- **Description:** Deploys a specifically profiled Apache Web Server (Port 80) honeypot.

### `DecoySSHD`
- **Module:** `deception.py`
- **Description:** Deploys a fake SSH daemon (Port 22) honeypot specifically designed to bait brute force actions.

### `DecoyTomcat`
- **Module:** `deception.py`
- **Description:** Deploys a fake Tomcat server (Port 8080) to deceive application port scans.

### `DeployDecoy`
- **Module:** `deception.py`
- **Description:** Deploys a generic high-interaction honeypot/decoy service to a target host.

### `DeployHoneytoken`
- **Module:** `deception.py`
- **Description:** Injects fake, highly-monitored credentials into the memory space of a real host. <br><br> If a Red agent successfully compromises this host and attempts to perform post-exploitation (e.g., Pass-the-Hash, credential dumping), they ingest the Honeytoken instead. This triggers an immediate, 100% confidence SIEM Alert exposing the Red agent's exact location natively.

### `Misinform`
- **Module:** `deception.py`
- **Description:** Injects false host telemetry or alters logging infrastructure to feed Red agents fake data.

### `RotateKerberos`
- **Module:** `identity.py`
- **Description:** Rotates Domain Kerberos TGT Keys globally to invalidate stolen Enterprise Admin tokens.

## Blue_operator Team Actions

### `Analyze`
- **Module:** `analysis.py`
- **Description:** Executes a forensic deep scan of a specific host for malware indicators.

### `Monitor`
- **Module:** `analysis.py`
- **Description:** Deploys active traffic analysis scanning on a specific subnet or host.

## Red Team Actions

### `DumpLSASS`
- **Module:** `post_exploitation.py`
- **Description:** Advanced Post-Exploitation Action: Scrapes memory for Active Directory tokens. Requires the Red Agent to have 'Root' privileges on the target Windows node.

### `ExploitBlueKeep`
- **Module:** `exploits.py`
- **Description:** Executes the CVE-2019-0708 (BlueKeep) vulnerability against Remote Desktop Services.

### `ExploitEternalBlue`
- **Module:** `exploits.py`
- **Description:** Executes the MS17-010 (EternalBlue) exploit targeting poorly configured SMBv1 servers.

### `ExploitHTTP_RFI`
- **Module:** `exploits.py`
- **Description:** Simulates a Remote File Inclusion (RFI) web application attack vector against Port 80/443.

### `ExploitRemoteService`
- **Module:** `exploits.py`
- **Description:** Attempts to weaponize a generic remote code execution vulnerability on a service. <br><br> If successful, upgrades the agent's standing from network access to local 'User' privileges on the target host. <br><br> Args:     agent_id (str): The unique identifier of the Red agent executing the exploit.     target_ip (str): The IPv4 address of the target server.     port (int, optional): The target TCP/UDP port mapping to the vulnerable service. Defaults to 80.

### `PassTheTicket`
- **Module:** `post_exploitation.py`
- **Description:** Lateral Movement via Identity validation bypassing CVE exploits explicitly.

### `PrivilegeEscalate`
- **Module:** `privilege_escalation.py`
- **Description:** Executes a generic local privilege escalation exploit on a compromised <br><br> host. <br><br> Elevates an agent's access from standard 'User' to 'Root' or 'SYSTEM', granting unrestricted control over the endpoint for subsequent impact actions. <br><br> Args:     agent_id (str): The unique identifier of the Red agent.     target_ip (str): The IP address of the already compromised host.

## Red_commander Team Actions

### `DiscoverNetworkServices`
- **Module:** `reconnaissance.py`
- **Description:** Executes an intrusive port scan against a specific host to enumerate <br><br> running daemons. <br><br> Often simulates an `nmap -sS -sV` scan to identify vulnerable service banners on open ports. <br><br> Args:     agent_id (str): The unique identifier of the Red agent.     target_ip (str): The IP address of the target host.

### `DiscoverRemoteSystems`
- **Module:** `reconnaissance.py`
- **Description:** Executes a targeted Ping Sweep against a subnet to explicitly identify <br><br> host machines. <br><br> This action simulates ICMP Echo Requests (Ping Sweeps) or ARP broadcasts to find live endpoints, susceptible to deception from Blue team honeypots. <br><br> Args:     agent_id (str): The unique identifier of the Red agent.     target_subnet (str): The CIDR target (e.g., "10.0.0.0/24").

### `NetworkScan`
- **Module:** `reconnaissance.py`
- **Description:** Executes a wide network scan across a specified subnet to map active IP <br><br> addresses. <br><br> This action represents the initial reconnaissance phase of the Cyber Kill Chain, typically mapping to MITRE ATT&CK T1046 (Network Service Scanning). <br><br> Args:     agent_id (str): The unique identifier of the Red agent executing the scan.     target_subnet (str): The CIDR block of the target subnet (e.g., "10.0.0.0/24").

### `ShareIntelligence`
- **Module:** `coordination.py`
- **Description:** Explicitly shares the current agent's 'Fog of War' knowledge graph with an allied agent.

## Red_operator Team Actions

### `ExfiltrateData`
- **Module:** `impact.py`
- **Description:** Exfiltrates sensitive data out of a compromised node.

### `Impact`
- **Module:** `impact.py`
- **Description:** Executes an impact objective (e.g., Ransomware/Wiper) to encrypt or destroy data on a compromised host.

### `JuicyPotato`
- **Module:** `privilege_escalation.py`
- **Description:** Simulates the JuicyPotato local privilege escalation vector leveraging <br><br> DCOM on Windows. <br><br> Abuses `SeImpersonatePrivilege` to elevate a service account to `NT AUTHORITY\SYSTEM`. <br><br> Args:     agent_id (str): Reference to the executing Red operator.     target_ip (str): Target IPv4 string.

### `KillProcess`
- **Module:** `impact.py`
- **Description:** Terminates a specific process (e.g., EDR sensor) on a compromised host.

### `OverloadPLC`
- **Module:** `kinetic.py`
- **Description:** Initiates a kinetic impact on a compromised Cyber-Physical OT Node to increase hardware temperatures.

### `PassTheHash`
- **Module:** `privilege_escalation.py`
- **Description:** Executes a lateral movement attack bypassing authentication using Kerberos / NTLM hashes extracted from a Domain Controller. <br><br> Args:     agent_id (str): Reference to the executing Red operator.     target_ip (str): Target IPv4 string (can be un-exploited if DC is cracked).

### `SpearPhishing`
- **Module:** `social_engineering.py`
- **Description:** Executes a targeted Social Engineering campaign against a Corporate End-User. <br><br> Unlike standard Exploits, SpearPhishing leverages email protocols and bypasses perimeter firewalls and DMZ routing constraints entirely. Its success probability is purely dictated by the `human_vulnerability_score` of the human operator assigned to the generated Endpoint, simulating clicks on malicious attachments. <br><br> Args:     agent_id (str): Reference ID of the Red operating unit.     target_ip (str): IP address of the target User Node (typically Corporate/Secure subnet).

### `V4L2KernelExploit`
- **Module:** `privilege_escalation.py`
- **Description:** Executes a specific kernel-level vulnerability via Video4Linux (V4L2) on <br><br> Linux targets. <br><br> Targets memory corruption in outdated kernel modules to spawn a root shell. <br><br> Args:     agent_id (str): Reference to the executing Red operator.     target_ip (str): Target IPv4 string.

