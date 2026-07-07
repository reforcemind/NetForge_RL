import json

import numpy as np
import pytest

from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.siem.export import export_ocsf, siem_to_ocsf


@pytest.mark.fast
def test_siem_to_ocsf_maps_event_id():
    log = (
        '<Event><System><EventID>4688</EventID>'
        '<Computer>10.0.0.5</Computer></System>'
        '<EventData><Data Name="CommandLine">whoami</Data></EventData></Event>'
    )
    rec = siem_to_ocsf(log, '10.0.0.0/24', tick=7)
    assert rec['class_name'] == 'Process Activity'
    assert rec['device']['hostname'] == '10.0.0.5'
    assert rec['time'] == 7
    assert rec['enrichments']['CommandLine'] == 'whoami'


@pytest.mark.fast
def test_incident_severity_is_elevated():
    rec = siem_to_ocsf('[INCIDENT] target=10.0.0.9 lateral movement', 'x', tick=1)
    assert rec['severity_id'] >= 6


@pytest.mark.fast
def test_export_requires_capture():
    env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 5})
    env.reset(seed=0)
    with pytest.raises(ValueError):
        export_ocsf(env, 'unused.jsonl')


@pytest.mark.integration
def test_export_writes_ocsf_jsonl(tmp_path):
    env = NetForgeRLEnv(
        {'scenario_type': 'ransomware', 'max_ticks': 40, 'record_siem': True}
    )
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    while env.agents:
        acts = {
            a: np.array([rng.integers(0, n) for n in env.action_space(a).nvec])
            for a in env.agents
        }
        _, _, term, trunc, _ = env.step(acts)
        if all(term.values()) or all(trunc.values()):
            break
    path = tmp_path / 'episode.ocsf.jsonl'
    n = export_ocsf(env, str(path))
    assert n > 0
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == n
    assert all('class_uid' in r and 'raw_event' in r for r in records)
