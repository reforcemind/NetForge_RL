# Scenarios

NetForge structures attacks and network defenses into parameterized scenario packs. Each scenario defines network topology, host roles, vulnerability configurations, adversary capabilities, and reward structures.

## Scenario Families

- **`ransomware`**: High-velocity opportunistic attacks. Red rapidly scans, compromises unpatched services, and encrypts filesystem assets. Blue must detect encryption early and isolate infected subnets before clinical or financial services halt.
- **`apt_espionage`**: Stealthy advanced persistent threat. Red avoids high-volume scanning, exploits zero-days or stolen credentials, dumps LSASS tokens, and exfiltrates sensitive records. Blue must correlate subtle authentication logs.
- **`cloud_hybrid`**: Hybrid cloud and enterprise infrastructure. Red leverages misconfigured IAM roles and cloud metadata services to pivot between on-premises workstations and cloud storage buckets.
- **`iot_fleet`**: High-density perimeter networks with resource-constrained IoT devices. Red builds botnet clusters to launch volumetric operations against core infrastructure.
- **`ot_stuxnet`**: Operational technology and industrial control systems (ICS). Red targets programmable logic controllers (PLCs), driving physical process setpoints into critical failure thresholds.

## Built-in Scenario Packs

Scenario packs are defined in YAML under `netforge_rl/scenarios/packs/`:

| Pack Name | Domain | Key Challenge |
|---|---|---|
| `hospital_ransomware` | Healthcare | High-availability clinical databases; severe SLA penalty for isolation |
| `enterprise_apt` | Corporate IT | Stealthy credential theft (Pass-the-Hash / Pass-the-Ticket) |
| `cloud_iam` | Cloud / Hybrid | Token credential leakage and cross-account privilege escalation |
| `finance` | Banking | Transaction ledger protection and compliance auditing |
| `iot_fleet` | Smart Infrastructure | Perimeter gateway compromise and distributed lateral movement |
| `ot_plant` | Industrial / SCADA | Physical sensor dynamics and PLC setpoint manipulation |
| `zero_trust` | Enterprise Identity | Ephemeral Kerberos ticket rotation and micro-segmentation |

## Running Scenario Packs

Run any scenario pack from the command line:

```bash
# Run a built-in pack
netforge run hospital_ransomware --seed 42 --max-ticks 100

# Run a custom scenario YAML file
netforge run my_scenario.yaml --replay replay.html
```

## Reward Design Variants

To evaluate agents under varying operational priorities, NetForge supports reward wrappers via `RewardDesignWrapper`:

- **`sla_only`**: Blue is scored purely on maintaining host availability and service uptime.
- **`security_only`**: Blue is scored purely on eradicating adversary compromises, regardless of host downtime.
- **`fp_penalized`**: Heavy penalties for isolating uncompromised machines (false positives).
- **`constrained_sla`**: Dual-objective optimization where security is maximized subject to minimum SLA bounds.
- **`safety_critical`**: Catastrophic penalties for kinetic failures (e.g., PLC destruction).
