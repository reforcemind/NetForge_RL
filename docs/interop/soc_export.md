# SOC Export & OCSF Schema

NetForge can stream and export telemetry using the [Open Cybersecurity Schema Framework (OCSF)](https://ocsf.io/), allowing simulation logs to be ingested into enterprise SIEM platforms like Splunk, Elastic, and Microsoft Sentinel.

## Enabling SIEM Recording

Set `record_siem=True` in environment configuration to capture the uncompressed event stream:

```python
from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.siem.export import export_ocsf

config = {
    "scenario_type": "enterprise_apt",
    "max_ticks": 100,
    "record_siem": True,
}

env = NetForgeRLEnv(config)
env.reset(seed=42)

# Run simulation steps...
for _ in range(50):
    actions = {agent: [0, 0] for agent in env.agents}
    env.step(actions)

# Export captured telemetry to OCSF JSONL format
export_ocsf(env, "runs/incident_trace.ocsf.jsonl")
```

## OCSF Schema Mapping

Simulation events map to standard OCSF event classes:

| Event Source | Internal Event ID | OCSF Class ID | OCSF Category |
|---|---|---|---|
| **Windows Security** | 4624 / 4625 / 4768 | `3002` | Authentication |
| **Windows Security** | 4688 / Sysmon 1 | `1007` | Process Activity |
| **Sysmon** | Event ID 3 | `4001` | Network Activity |
| **Sysmon** | Event ID 22 | `4003` | DNS Query |
| **SCADA Physics** | Physical Alarm | `2001` | System Activity (Security Finding) |

Raw event details are preserved in the `raw_event` field, allowing SOC engineers to test correlation rules against realistic synthetic data.
