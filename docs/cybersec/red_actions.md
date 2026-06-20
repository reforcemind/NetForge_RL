# Red Team Actions

### `DumpLSASS`
- **Module:** `post_exploitation.py`
- **Description:** Extracts token arrays from memory. Requires `Root` privilege state on the target node.

### `ExploitBlueKeep` / `ExploitEternalBlue` / `ExploitHTTP_RFI`
- **Module:** `exploits.py`
- **Description:** Target specific service vulnerabilities (RDP, SMB, HTTP). Validates against host vulnerability vectors to modify access state.

### `ExploitRemoteService`
- **Module:** `exploits.py`
- **Description:** Generic exploit attempting to modify access state from network visibility to local `User` privilege.
- **Args:** `agent_id` (str), `target_ip` (str), `port` (int).

### `PassTheTicket` / `PassTheHash`
- **Module:** `post_exploitation.py` / `privilege_escalation.py`
- **Description:** Validates previously extracted token arrays to modify access state on target nodes without requiring vulnerable services.

### `PrivilegeEscalate`
- **Module:** `privilege_escalation.py`
- **Description:** Modifies agent privilege state from `User` to `Root`/`SYSTEM`.
- **Args:** `agent_id` (str), `target_ip` (str).

### `DiscoverNetworkServices`
- **Module:** `reconnaissance.py`
- **Description:** Queries the active service array of a target node. Updates agent visibility mask.
- **Args:** `agent_id` (str), `target_ip` (str).

### `DiscoverRemoteSystems` / `NetworkScan`
- **Module:** `reconnaissance.py`
- **Description:** Queries the active IP allocation array of a target subnet. Updates agent visibility mask.
- **Args:** `agent_id` (str), `target_subnet` (str).

### `ShareIntelligence`
- **Module:** `coordination.py`
- **Description:** Performs a union operation on the visibility masks of two allied agents.

### `ExfiltrateData` / `Impact` / `KillProcess`
- **Module:** `impact.py`
- **Description:** Executes terminal objectives. Modifies the data state or process array on a compromised node to calculate scenario rewards.

### `JuicyPotato` / `V4L2KernelExploit`
- **Module:** `privilege_escalation.py`
- **Description:** Platform-specific privilege escalation actions targeting specific OS vectors (Windows DCOM / Linux Kernel).

### `OverloadPLC`
- **Module:** `kinetic.py`
- **Description:** Modifies the physical state tensor of an OT node to `kinetic_destruction`.

### `SpearPhishing`
- **Module:** `social_engineering.py`
- **Description:** Evaluates success strictly against the `human_vulnerability_score` scalar of a target endpoint. Bypasses routing graph constraints.
- **Args:** `agent_id` (str), `target_ip` (str).
