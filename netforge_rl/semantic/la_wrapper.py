"""Language-Action wrapper: frozen ``EnvState`` -> SIEM-style text report.

Reports are templated per agent role (Red operator vs Blue SOC). The
classifier reuses the renderer's color taxonomy so the text and image
modalities stay in lock-step.
"""

from __future__ import annotations

from netforge_rl.core.functional import (
    DECOY_CODES,
    EnvState,
    PRIVILEGE_CODES,
    STATUS_CODES,
)
from netforge_rl.semantic.action_menu import action_menu


def _is_padding(subnet: str) -> bool:
    return subnet.startswith('169.254.')


def _host_phrase(state: EnvState, idx: int) -> str:
    ip = state.meta.ip[idx]
    status = STATUS_CODES[int(state.hosts.status[idx])]
    priv = PRIVILEGE_CODES[int(state.hosts.privilege[idx])]
    by = int(state.hosts.compromised_by_id[idx])
    actor = state.agent_ids[by] if by >= 0 else None
    parts = [f'{ip} ({state.meta.hostname[idx]})']
    if status != 'online':
        parts.append(status.upper())
    if priv != 'None':
        parts.append(f'priv={priv}')
    if actor:
        parts.append(f'owned by {actor}')
    if bool(state.hosts.contains_honeytokens[idx]):
        parts.append('HONEYTOKEN')
    if DECOY_CODES[int(state.hosts.decoy[idx])] != 'inactive':
        parts.append(f'DECOY={DECOY_CODES[int(state.hosts.decoy[idx])]}')
    return ' | '.join(parts)


def state_to_text(
    state: EnvState,
    agent_id: str,
    *,
    max_hosts: int = 12,
    include_menu: bool = True,
) -> str:
    """Render a SIEM-style telemetry report for ``agent_id``.

    ``max_hosts`` caps the most-interesting-first host list so the report
    fits inside small context windows. Padding hosts are filtered.
    """
    role = 'Blue SOC Operator' if 'blue' in agent_id.lower() else 'Red Operator'
    j = state.agent_ids.index(agent_id) if agent_id in state.agent_ids else None

    # Triage: surface non-default hosts first.
    interesting: list[int] = []
    others: list[int] = []
    for i, sn in enumerate(state.meta.subnet_cidr):
        if _is_padding(sn):
            continue
        non_default = (
            STATUS_CODES[int(state.hosts.status[i])] != 'online'
            or PRIVILEGE_CODES[int(state.hosts.privilege[i])] != 'None'
            or int(state.hosts.compromised_by_id[i]) >= 0
            or bool(state.hosts.contains_honeytokens[i])
            or bool(state.hosts.edr_active[i])
        )
        (interesting if non_default else others).append(i)
    shown = (interesting + others)[:max_hosts]
    n_active = len(interesting) + len(others)

    lines = [
        f'[NetForge SIEM Report — tick {state.current_tick}]',
        f'Agent: {agent_id}  Role: {role}',
        f'Active hosts: {n_active}  Showing top {len(shown)}',
    ]
    if j is not None:
        lines.append(
            f'Budgets — energy={int(state.agent_energy[j])} '
            f'funds={int(state.agent_funds[j])} '
            f'compute={int(state.agent_compute[j])}'
        )
    lines.append('Telemetry:')
    for i in shown:
        lines.append(f'  - {_host_phrase(state, i)}')

    if include_menu:
        menu = action_menu(agent_id)
        lines.append('Legal actions (action_id: name):')
        for gid, name in menu.items():
            lines.append(f'  {gid}: {name}')
        lines.append(
            'Reply format: `ACTION <action_id> TARGET <host_ip>` '
            '(use one of the IPs above; index will be inferred).'
        )

    return '\n'.join(lines)
