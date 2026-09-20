# SOC export (OCSF)

`record_siem=True` keeps the full stream. `export_ocsf` writes JSONL.

```python
from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.siem.export import export_ocsf, siem_to_ocsf

env = NetForgeRLEnv(
    {'scenario_type': 'ransomware', 'max_ticks': 100, 'record_siem': True}
)
env.reset(seed=0)
export_ocsf(env, 'runs/episode.ocsf.jsonl')
siem_to_ocsf(log_line, subnet='10.0.0.0/24', tick=42)
```

| EventID | OCSF |
|---|---|
| 4624 / 4625 / 4648 / 4768 / 4776 | Authentication (3002) |
| 4688 / Sysmon 1 / 10 | Process (1007) |
| Sysmon 3 | Network (4001) |
| Sysmon 22 | DNS (4003) |

Raw XML stays in `raw_event`. Honeytoken / incident → higher `severity_id`.
