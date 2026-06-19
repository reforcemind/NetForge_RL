# Blue Team Actions

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

