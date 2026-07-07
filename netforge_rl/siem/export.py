from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

# Re-maps NetForge's Windows/Sysmon event XML into OCSF-style JSON records.
_EVENTID = re.compile(r'<EventID>(\d+)</EventID>')
_COMPUTER = re.compile(r'<Computer>(.*?)</Computer>')
_DATA = re.compile(r'<Data Name="(.*?)">(.*?)</Data>', re.DOTALL)

# EventID -> (OCSF class_uid, class_name, activity)
_OCSF_MAP = {
    '4624': (3002, 'Authentication', 'Logon'),
    '4625': (3002, 'Authentication', 'Logon Failure'),
    '4648': (3002, 'Authentication', 'Explicit Credential Logon'),
    '4688': (1007, 'Process Activity', 'Launch'),
    '4768': (3002, 'Authentication', 'Kerberos TGT Request'),
    '4776': (3002, 'Authentication', 'NTLM Validation'),
    '1': (1007, 'Process Activity', 'Process Creation'),
    '3': (4001, 'Network Activity', 'Network Connection'),
    '10': (1007, 'Process Activity', 'Process Access'),
    '22': (4003, 'DNS Activity', 'DNS Query'),
}


def siem_to_ocsf(log_line: str, subnet: str, tick: Optional[int] = None) -> dict:
    """Map one NetForge SIEM log line to an OCSF-style event record."""
    tags = [t.strip('[]') for t in re.findall(r'\[[A-Z_]+\]', log_line)]
    eid_match = _EVENTID.search(log_line)
    event_id = eid_match.group(1) if eid_match else None
    class_uid, class_name, activity = _OCSF_MAP.get(
        event_id, (0, 'Uncategorized', 'Unknown')
    )
    computer = _COMPUTER.search(log_line)
    fields = {k: v.strip() for k, v in _DATA.findall(log_line)}

    severity = 1
    if 'INCIDENT' in tags or 'HONEYTOKEN_TRIGGERED' in log_line:
        severity = 6
    elif event_id == '10' or 'mimikatz' in log_line.lower():
        severity = 4

    return {
        'metadata': {
            'product': {'name': 'NetForge RL', 'vendor_name': 'ReforceMind'},
            'version': '1.1.0',
            'original_event_id': event_id,
            'labels': tags,
        },
        'class_uid': class_uid,
        'class_name': class_name,
        'activity_name': activity,
        'time': tick,
        'severity_id': severity,
        'device': {
            'hostname': computer.group(1) if computer else None,
            'subnet': subnet,
        },
        'enrichments': fields,
        'raw_event': log_line,
    }


def to_ocsf_records(entries: Iterable[tuple], *, has_tick: bool = True) -> list[dict]:
    """Convert entries to OCSF records. With has_tick, entries are
    (tick, line, subnet); otherwise (line, subnet)."""
    records = []
    for entry in entries:
        if has_tick:
            tick, log_line, subnet = entry
        else:
            tick, (log_line, subnet) = None, entry
        records.append(siem_to_ocsf(log_line, subnet, tick))
    return records


def export_ocsf(env, path: str) -> int:
    """Write the env's captured SIEM stream (record_siem=True) as OCSF JSONL. Returns
    the event count."""
    captured = getattr(env.siem_logger, 'captured', None)
    if captured is None:
        raise ValueError(
            'Environment was not built with record_siem=True; nothing to export.'
        )
    records = to_ocsf_records(captured, has_tick=True)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(json.dumps(r) for r in records))
    return len(records)
