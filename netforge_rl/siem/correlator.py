from typing import List
from netforge_rl.core.state import GlobalNetworkState


_STAGE_KEYWORDS = {
    'RECON': [
        'port_scan',
        'ping_sweep',
        'DiscoverNetworkServices',
        'discovered_subnet',
    ],
    'EXPLOIT': [
        'exploit_payload',
        'ExploitEternalBlue',
        'ExploitBlueKeep',
        'ExploitHTTP_RFI',
        'BlueKeep',
        'EternalBlue',
        'User_Access',
    ],
    'PRIVESC': ['JuicyPotato', 'V4L2', 'PrivilegeEscalate', 'Root', 'elevated'],
    'LATERAL': ['PassTheHash', 'PassTheTicket', 'DumpLSASS', 'lateral', 'pivot'],
    'IMPACT': [
        'exfil',
        'ransomware',
        'KillProcess',
        'SCADA_PHYSICAL',
        'HONEYTOKEN',
        'kinetic',
        'Impact',
    ],
}

_STAGE_ORDER = list(_STAGE_KEYWORDS)


class SIEMCorrelator:
    """Aggregates raw SIEM log buffer into typed incident records."""

    def __init__(self):
        self._last_correlated_tick = -1

    def reset(self) -> None:
        self._last_correlated_tick = -1

    def correlate(self, global_state: 'GlobalNetworkState') -> List[str]:
        """Returns incident log strings for this tick. Call once per step."""
        if global_state.current_tick == self._last_correlated_tick:
            return []
        self._last_correlated_tick = global_state.current_tick

        incidents = []
        seen_entries = [
            (log, subnet)
            for log, subnet in global_state.siem_log_buffer
            if isinstance(log, str)
        ]

        ip_stages: dict = {}
        for log, subnet in seen_entries:
            for stage, keywords in _STAGE_KEYWORDS.items():
                if any(kw.lower() in log.lower() for kw in keywords):
                    for token in log.split():
                        if token.count('.') == 3:
                            ip_stages.setdefault(token, set()).add(stage)
                            break
        for ip, host in global_state.all_hosts.items():
            if host.privilege in ('User', 'Root'):
                ip_stages.setdefault(ip, set()).add('EXPLOIT')
            if host.privilege == 'Root':
                ip_stages.setdefault(ip, set()).add('PRIVESC')

        for ip, stages in ip_stages.items():
            if not stages:
                continue
            highest = max(
                (_STAGE_ORDER.index(s) for s in stages if s in _STAGE_ORDER),
                default=0,
            )
            stage_name = _STAGE_ORDER[highest]
            confidence = min(len(stages) / len(_STAGE_ORDER), 1.0)
            host = global_state.all_hosts.get(ip)
            subnet = host.subnet_cidr if host else 'unknown'

            if len(stages) >= 2 or confidence >= 0.4:
                incidents.append(
                    (
                        f'[INCIDENT] stage={stage_name} target={ip} '
                        f'confidence={confidence:.2f} '
                        f'stages_observed={",".join(sorted(stages))} '
                        f'tick={global_state.current_tick}',
                        subnet,
                    )
                )

        return incidents
