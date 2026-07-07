from __future__ import annotations

import datetime
import random
from typing import Callable

_rng = random.Random()

_BASE_TIME = datetime.datetime(2026, 1, 1, 0, 0, 0)


def seed_events(seed: int | None) -> None:
    """Reseed the fallback template RNG; call once per episode reset."""
    _rng.seed(seed)


def _resolve_rng(kw: dict) -> random.Random:
    """Per-call RNG from kwargs, falling back to the module global."""
    return kw.get('rng') or _rng


def _ts(rng: random.Random | None = None) -> str:
    """Return a deterministic, plausible timestamp string."""
    rng = rng or _rng
    jitter = datetime.timedelta(
        hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59),
        seconds=rng.randint(0, 59),
    )
    return (_BASE_TIME - jitter).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def evid_4624(src_ip: str, target_ip: str, username: str = 'SYSTEM', **kw) -> str:
    """4624 — An account was successfully logged on."""
    rng = _resolve_rng(kw)
    return (
        f'<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">\n'
        f'  <System><EventID>4624</EventID><TimeCreated SystemTime="{_ts(rng)}"/>'
        f'<Computer>{target_ip}</Computer></System>\n'
        f'  <EventData>\n'
        f'    <Data Name="SubjectUserName">{username}</Data>\n'
        f'    <Data Name="LogonType">3</Data>\n'
        f'    <Data Name="IpAddress">{src_ip}</Data>\n'
        f'    <Data Name="AuthenticationPackageName">NTLM</Data>\n'
        f'    <Data Name="WorkstationName">{src_ip}</Data>\n'
        f'  </EventData>\n'
        f'</Event>'
    )


def evid_4625(
    src_ip: str, target_ip: str, username: str = 'Administrator', **kw
) -> str:
    """4625 — An account failed to log on."""
    rng = _resolve_rng(kw)
    failure_reasons = ['%%2313', '%%2304', '%%2308']
    return (
        f'<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">\n'
        f'  <System><EventID>4625</EventID><TimeCreated SystemTime="{_ts(rng)}"/>'
        f'<Computer>{target_ip}</Computer></System>\n'
        f'  <EventData>\n'
        f'    <Data Name="TargetUserName">{username}</Data>\n'
        f'    <Data Name="LogonType">3</Data>\n'
        f'    <Data Name="IpAddress">{src_ip}</Data>\n'
        f'    <Data Name="FailureReason">{rng.choice(failure_reasons)}</Data>\n'
        f'    <Data Name="Status">0xC000006D</Data>\n'
        f'  </EventData>\n'
        f'</Event>'
    )


def evid_4648(
    src_ip: str, target_ip: str, username: str = 'Administrator', **kw
) -> str:
    """4648 — A logon was attempted using explicit credentials (Pass-the-Hash indicator)."""
    rng = _resolve_rng(kw)
    return (
        f'<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">\n'
        f'  <System><EventID>4648</EventID><TimeCreated SystemTime="{_ts(rng)}"/>'
        f'<Computer>{src_ip}</Computer></System>\n'
        f'  <EventData>\n'
        f'    <Data Name="SubjectUserName">SYSTEM</Data>\n'
        f'    <Data Name="TargetUserName">{username}</Data>\n'
        f'    <Data Name="TargetServerName">{target_ip}</Data>\n'
        f'    <Data Name="ProcessName">C:\\Windows\\System32\\lsass.exe</Data>\n'
        f'  </EventData>\n'
        f'</Event>'
    )


def evid_4688(
    src_ip: str, process: str = 'cmd.exe', parent: str = 'explorer.exe', **kw
) -> str:
    """4688 — A new process has been created."""
    rng = _resolve_rng(kw)
    cmdlines = {
        'cmd.exe': 'C:\\Windows\\system32\\cmd.exe /c whoami',
        'powershell.exe': 'powershell.exe -NoP -NonI -W Hidden -Exec Bypass -Enc <base64>',
        'net.exe': 'net user /domain',
        'mimikatz.exe': 'mimikatz.exe privilege::debug sekurlsa::logonpasswords exit',
        'procdump.exe': 'procdump.exe -ma lsass.exe lsass.dmp',
    }
    cmdline = cmdlines.get(process, f'{process} --help')
    return (
        f'<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">\n'
        f'  <System><EventID>4688</EventID><TimeCreated SystemTime="{_ts(rng)}"/>'
        f'<Computer>{src_ip}</Computer></System>\n'
        f'  <EventData>\n'
        f'    <Data Name="NewProcessName">C:\\Windows\\System32\\{process}</Data>\n'
        f'    <Data Name="ParentProcessName">C:\\Windows\\{parent}</Data>\n'
        f'    <Data Name="CommandLine">{cmdline}</Data>\n'
        f'    <Data Name="SubjectUserName">NT AUTHORITY\\SYSTEM</Data>\n'
        f'  </EventData>\n'
        f'</Event>'
    )


def evid_4768(
    src_ip: str, target_ip: str, username: str = 'Administrator', **kw
) -> str:
    """4768 — A Kerberos authentication ticket (TGT) was requested."""
    rng = _resolve_rng(kw)
    return (
        f'<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">\n'
        f'  <System><EventID>4768</EventID><TimeCreated SystemTime="{_ts(rng)}"/>'
        f'<Computer>{target_ip}</Computer></System>\n'
        f'  <EventData>\n'
        f'    <Data Name="TargetUserName">{username}</Data>\n'
        f'    <Data Name="TargetDomainName">CORP</Data>\n'
        f'    <Data Name="IpAddress">::ffff:{src_ip}</Data>\n'
        f'    <Data Name="Status">0x0</Data>\n'
        f'    <Data Name="TicketEncryptionType">0x12</Data>\n'
        f'  </EventData>\n'
        f'</Event>'
    )


def evid_4776(
    src_ip: str, target_ip: str, username: str = 'Administrator', **kw
) -> str:
    """4776 — The computer attempted to validate credentials for an account (NTLM auth)."""
    rng = _resolve_rng(kw)
    return (
        f'<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">\n'
        f'  <System><EventID>4776</EventID><TimeCreated SystemTime="{_ts(rng)}"/>'
        f'<Computer>{target_ip}</Computer></System>\n'
        f'  <EventData>\n'
        f'    <Data Name="PackageName">MICROSOFT_AUTHENTICATION_PACKAGE_V1_0</Data>\n'
        f'    <Data Name="TargetUserName">{username}</Data>\n'
        f'    <Data Name="Workstation">{src_ip}</Data>\n'
        f'    <Data Name="Status">0x0</Data>\n'
        f'  </EventData>\n'
        f'</Event>'
    )


def sysmon_1(src_ip: str, process: str = 'powershell.exe', **kw) -> str:
    """Sysmon Event ID 1 — Process Creation."""
    rng = _resolve_rng(kw)
    hashes = {
        'powershell.exe': 'SHA256=A8FDBA9DF15E41B6F5C69C79F66A94770913A498',
        'cmd.exe': 'SHA256=B99D61D874728EDC0918CA0EB10EAB93D381E7367E377406',
        'mimikatz.exe': 'SHA256=D4A0FE56316A2C45771BDE22F6F7AB40C59A6FBE',
    }
    hsh = hashes.get(process, 'SHA256=UNKNOWN')
    return (
        f'<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">\n'
        f'  <System><Provider Name="Microsoft-Windows-Sysmon"/>'
        f'<EventID>1</EventID><TimeCreated SystemTime="{_ts(rng)}"/>'
        f'<Computer>{src_ip}</Computer></System>\n'
        f'  <EventData>\n'
        f'    <Data Name="Image">C:\\Windows\\System32\\{process}</Data>\n'
        f'    <Data Name="Hashes">{hsh}</Data>\n'
        f'    <Data Name="ParentImage">C:\\Windows\\explorer.exe</Data>\n'
        f'    <Data Name="User">NT AUTHORITY\\SYSTEM</Data>\n'
        f'    <Data Name="IntegrityLevel">System</Data>\n'
        f'  </EventData>\n'
        f'</Event>'
    )


def sysmon_3(src_ip: str, target_ip: str, dst_port: int = 445, **kw) -> str:
    """Sysmon Event ID 3 — Network Connection Detected."""
    rng = _resolve_rng(kw)
    return (
        f'<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">\n'
        f'  <System><Provider Name="Microsoft-Windows-Sysmon"/>'
        f'<EventID>3</EventID><TimeCreated SystemTime="{_ts(rng)}"/>'
        f'<Computer>{src_ip}</Computer></System>\n'
        f'  <EventData>\n'
        f'    <Data Name="Image">C:\\Windows\\System32\\svchost.exe</Data>\n'
        f'    <Data Name="SourceIp">{src_ip}</Data>\n'
        f'    <Data Name="SourcePort">{rng.randint(49152, 65535)}</Data>\n'
        f'    <Data Name="DestinationIp">{target_ip}</Data>\n'
        f'    <Data Name="DestinationPort">{dst_port}</Data>\n'
        f'    <Data Name="Protocol">tcp</Data>\n'
        f'  </EventData>\n'
        f'</Event>'
    )


def sysmon_10(src_ip: str, **kw) -> str:
    """Sysmon Event ID 10 — ProcessAccess (LSASS credentialdumping indicator)."""
    rng = _resolve_rng(kw)
    return (
        f'<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">\n'
        f'  <System><Provider Name="Microsoft-Windows-Sysmon"/>'
        f'<EventID>10</EventID><TimeCreated SystemTime="{_ts(rng)}"/>'
        f'<Computer>{src_ip}</Computer></System>\n'
        f'  <EventData>\n'
        f'    <Data Name="SourceImage">C:\\Windows\\System32\\cmd.exe</Data>\n'
        f'    <Data Name="TargetImage">C:\\Windows\\System32\\lsass.exe</Data>\n'
        f'    <Data Name="GrantedAccess">0x1010</Data>\n'
        f'    <Data Name="CallTrace">C:\\Windows\\SYSTEM32\\ntdll.dll+...</Data>\n'
        f'  </EventData>\n'
        f'</Event>'
    )


def sysmon_22(src_ip: str, domain: str = 'corp.internal', **kw) -> str:
    """Sysmon Event ID 22 — DNS Query."""
    rng = _resolve_rng(kw)
    queries = [
        f'dc01.{domain}',
        f'ldap.{domain}',
        f'kerberos.{domain}',
        f'krbtgt.{domain}',
        'time.windows.com',
    ]
    return (
        f'<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">\n'
        f'  <System><Provider Name="Microsoft-Windows-Sysmon"/>'
        f'<EventID>22</EventID><TimeCreated SystemTime="{_ts(rng)}"/>'
        f'<Computer>{src_ip}</Computer></System>\n'
        f'  <EventData>\n'
        f'    <Data Name="QueryName">{rng.choice(queries)}</Data>\n'
        f'    <Data Name="QueryResults">type: 1 {src_ip}</Data>\n'
        f'    <Data Name="Image">C:\\Windows\\System32\\lsass.exe</Data>\n'
        f'  </EventData>\n'
        f'</Event>'
    )


ActionEventMap = dict[str, list[tuple[float, Callable]]]

ACTION_EVENT_MAP: ActionEventMap = {
    'ExploitEternalBlue': [
        (0.5, lambda s, t, **kw: sysmon_3(s, t, dst_port=445, **kw)),
        (0.3, lambda s, t, **kw: evid_4624(s, t, username='Administrator', **kw)),
        (0.2, lambda s, t, **kw: sysmon_1(s, process='cmd.exe', **kw)),
    ],
    'ExploitBlueKeep': [
        (0.6, lambda s, t, **kw: sysmon_3(s, t, dst_port=3389, **kw)),
        (0.4, lambda s, t, **kw: evid_4624(s, t, username='Administrator', **kw)),
    ],
    'ExploitHTTP_RFI': [
        (0.5, lambda s, t, **kw: sysmon_3(s, t, dst_port=80, **kw)),
        (0.3, lambda s, t, **kw: sysmon_1(s, process='cmd.exe', **kw)),
        (
            0.2,
            lambda s, t, **kw: evid_4688(
                s, process='php-cgi.exe', parent='httpd.exe', **kw
            ),
        ),
    ],
    'ExploitRemoteService': [
        (0.4, lambda s, t, **kw: sysmon_3(s, t, dst_port=22, **kw)),
        (0.4, lambda s, t, **kw: evid_4625(s, t, **kw)),
        (0.2, lambda s, t, **kw: sysmon_1(s, process='bash', **kw)),
    ],
    'PrivilegeEscalate': [
        (0.5, lambda s, t, **kw: evid_4688(s, process='cmd.exe', **kw)),
        (0.5, lambda s, t, **kw: sysmon_1(s, process='powershell.exe', **kw)),
    ],
    'DumpLSASS': [
        (0.5, lambda s, t, **kw: sysmon_10(s, **kw)),
        (0.3, lambda s, t, **kw: evid_4688(s, process='mimikatz.exe', **kw)),
        (0.2, lambda s, t, **kw: evid_4688(s, process='procdump.exe', **kw)),
    ],
    'PassTheTicket': [
        (0.5, lambda s, t, **kw: evid_4768(s, t, **kw)),
        (0.3, lambda s, t, **kw: evid_4648(s, t, **kw)),
        (0.2, lambda s, t, **kw: evid_4776(s, t, **kw)),
    ],
    'NetworkScan': [
        (
            0.6,
            lambda s, t, **kw: sysmon_3(
                s, t, dst_port=_resolve_rng(kw).choice([22, 80, 443, 445]), **kw
            ),
        ),
        (0.4, lambda s, t, **kw: evid_4625(s, t, **kw)),
    ],
    'IsolateHost': [
        (
            1.0,
            lambda s, t, **kw: evid_4688(
                s, process='netsh.exe', parent='services.exe', **kw
            ),
        ),
    ],
    'RotateKerberos': [
        (0.6, lambda s, t, **kw: evid_4768(s, t, **kw)),
        (0.4, lambda s, t, **kw: evid_4776(s, t, **kw)),
    ],
    '_default': [
        (0.5, lambda s, t, **kw: sysmon_3(s, t, **kw)),
        (0.5, lambda s, t, **kw: evid_4688(s, process='cmd.exe', **kw)),
    ],
}
