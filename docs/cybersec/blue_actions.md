# Blue Team Actions

### `ConfigureACL`
- **Module:** `mitigation.py`
- **Description:** Modifies the firewall state to drop traffic targeting specific ports on a subnet.

### `IsolateHost`
- **Module:** `mitigation.py`
- **Description:** Sets target host status to isolated, dropping all inbound and outbound traffic capabilities.

### `Remove`
- **Module:** `mitigation.py`
- **Description:** Resets the compromised state of a target host, terminating executing Red policies on that node.

### `RestoreFromBackup`
- **Module:** `mitigation.py`
- **Description:** Reverts target host state vector to baseline initial configuration.

### `RestoreHost`
- **Module:** `mitigation.py`
- **Description:** Reverts host status from isolated, re-establishing network connectivity.

### `SecurityAwarenessTraining`
- **Module:** `mitigation.py`
- **Description:** Decreases the `human_vulnerability_score` scalar for all nodes in a target subnet.

### `DecoyApache` / `DecoySSHD` / `DecoyTomcat`
- **Module:** `deception.py`
- **Description:** Instantiates a decoy service node matching specific port signatures (80, 22, 8080) to intercept discovery requests.

### `DeployDecoy`
- **Module:** `deception.py`
- **Description:** Instantiates a generic decoy service node.

### `DeployHoneytoken`
- **Module:** `deception.py`
- **Description:** Injects a honeytoken state into a host. Subsequent token extraction actions by Red agents trigger an explicit SIEM event with perfect confidence mapping to the executor's ID.

### `Misinform`
- **Module:** `deception.py`
- **Description:** Alters telemetry states to return falsified observation arrays to Red agent discovery actions.

### `RotateKerberos`
- **Module:** `identity.py`
- **Description:** Invalidates existing Domain Controller token arrays globally.

### `Analyze`
- **Module:** `analysis.py`
- **Description:** Queries the complete unmasked state of a specific host node.

### `Monitor`
- **Module:** `analysis.py`
- **Description:** Increases the SIEM event generation probability scalar for actions executing on a target subnet.
