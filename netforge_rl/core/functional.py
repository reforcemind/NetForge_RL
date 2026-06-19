from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING
import numpy as np

N_HOSTS = 100
STATUS_CODES = ('online', 'isolated', 'kernel_panic')
PRIVILEGE_CODES = ('None', 'User', 'Root')
DECOY_CODES = ('inactive', 'active', 'Apache', 'SSHD', 'Tomcat')
INTEGRITY_CODES = ('clean', 'compromised', 'kinetic_destruction')
CVE_CODES = (
    'MS17-010',
    'CVE-2019-0708',
    'CVE-2021-44228',
    'V4L2',
    'CVE-2010-2772',
    'Stuxnet_0day',
)
N_CVE = len(CVE_CODES)
TOKEN_CODES = ('Enterprise_Admin_Token', 'Local_Admin_DMZ', 'Local_Admin_Corporate')
N_TOKEN = len(TOKEN_CODES)
OS_OTHER, OS_WINDOWS, OS_LINUX, OS_PLC = (0, 1, 2, 3)


def _os_family_code(os_str):
    s = str(os_str or '')
    if 'Windows' in s:
        return OS_WINDOWS
    if 'Linux' in s:
        return OS_LINUX
    if 'PLC' in s:
        return OS_PLC
    return OS_OTHER


def _encode(value, codebook):
    try:
        return codebook.index(value)
    except ValueError:
        return 0


def _decode(code, codebook):
    code = int(code)
    if 0 <= code < len(codebook):
        return codebook[code]
    return codebook[0]


@dataclass(frozen=True)
class HostArrays:
    """Vectorizable per-host SoA — JAX PyTree leaves."""

    status: np.ndarray
    privilege: np.ndarray
    decoy: np.ndarray
    edr_active: np.ndarray
    is_domain_controller: np.ndarray
    contains_honeytokens: np.ndarray
    human_vulnerability: np.ndarray
    cvss_score: np.ndarray
    compromised_by_id: np.ndarray
    system_integrity: np.ndarray
    vuln_mask: np.ndarray
    host_tokens: np.ndarray
    os_family: np.ndarray


@dataclass(frozen=True)
class HostMeta:
    """Static / variable-length host metadata. Not a PyTree leaf."""

    ip: tuple
    hostname: tuple
    subnet_cidr: tuple
    os: tuple
    services: tuple
    vulnerabilities: tuple
    cached_credentials: tuple
    system_tokens: tuple


@dataclass(frozen=True)
class EnvState:
    """Immutable snapshot of the MARL environment."""

    hosts: HostArrays
    meta: HostMeta
    agent_ids: tuple
    agent_energy: np.ndarray
    agent_funds: np.ndarray
    agent_compute: np.ndarray
    agent_locked_until: np.ndarray
    current_tick: int = 0
    business_downtime_score: float = 0.0
    knowledge: tuple = field(default_factory=tuple)
    inventory: tuple = field(default_factory=tuple)

    @property
    def host_count(self):
        return self.hosts.status.shape[0]

    def agent_index(self, agent_id):
        return self.agent_ids.index(agent_id)

    def host_index(self, ip):
        return self.meta.ip.index(ip)

    def with_tick(self, tick):
        return replace(self, current_tick=tick)


def from_global_state(legacy, agent_ids):
    """Build a frozen EnvState from a legacy GlobalNetworkState."""
    sorted_ips = tuple(sorted(legacy.all_hosts.keys()))
    n = len(sorted_ips)
    if n != N_HOSTS:
        raise ValueError(f'Expected exactly {N_HOSTS} hosts; got {n}.')
    hosts_in_order = [legacy.all_hosts[ip] for ip in sorted_ips]
    status_arr = np.array(
        [_encode(h.status, STATUS_CODES) for h in hosts_in_order], dtype=np.int8
    )
    priv_arr = np.array(
        [_encode(h.privilege, PRIVILEGE_CODES) for h in hosts_in_order], dtype=np.int8
    )
    decoy_arr = np.array(
        [_encode(h.decoy, DECOY_CODES) for h in hosts_in_order], dtype=np.int8
    )
    edr_arr = np.array([bool(h.edr_active) for h in hosts_in_order], dtype=bool)
    dc_arr = np.array(
        [bool(h.is_domain_controller) for h in hosts_in_order], dtype=bool
    )
    honey_arr = np.array(
        [bool(getattr(h, 'contains_honeytokens', False)) for h in hosts_in_order],
        dtype=bool,
    )
    hvuln_arr = np.array(
        [float(h.human_vulnerability_score) for h in hosts_in_order], dtype=np.float32
    )
    cvss_arr = np.array(
        [float(getattr(h, 'cvss_score', 0.0)) for h in hosts_in_order], dtype=np.float32
    )
    comp_arr = np.array(
        [
            agent_ids.index(h.compromised_by) if h.compromised_by in agent_ids else -1
            for h in hosts_in_order
        ],
        dtype=np.int8,
    )
    integrity_arr = np.array(
        [
            _encode(getattr(h, 'system_integrity', 'clean'), INTEGRITY_CODES)
            for h in hosts_in_order
        ],
        dtype=np.int8,
    )
    vuln_mask = np.zeros((n, N_CVE), dtype=bool)
    for i, h in enumerate(hosts_in_order):
        for cve in getattr(h, 'vulnerabilities', None) or ():
            if cve in CVE_CODES:
                vuln_mask[i, CVE_CODES.index(cve)] = True
    host_tokens = np.zeros((n, N_TOKEN), dtype=bool)
    for i, h in enumerate(hosts_in_order):
        for tok in getattr(h, 'cached_credentials', None) or ():
            if tok in TOKEN_CODES:
                host_tokens[i, TOKEN_CODES.index(tok)] = True
        for tok in getattr(h, 'system_tokens', None) or ():
            if tok in TOKEN_CODES:
                host_tokens[i, TOKEN_CODES.index(tok)] = True
    os_family = np.array([_os_family_code(h.os) for h in hosts_in_order], dtype=np.int8)
    hosts = HostArrays(
        status=status_arr,
        privilege=priv_arr,
        decoy=decoy_arr,
        edr_active=edr_arr,
        is_domain_controller=dc_arr,
        contains_honeytokens=honey_arr,
        human_vulnerability=hvuln_arr,
        cvss_score=cvss_arr,
        compromised_by_id=comp_arr,
        system_integrity=integrity_arr,
        vuln_mask=vuln_mask,
        host_tokens=host_tokens,
        os_family=os_family,
    )
    meta = HostMeta(
        ip=sorted_ips,
        hostname=tuple((h.hostname for h in hosts_in_order)),
        subnet_cidr=tuple((h.subnet_cidr for h in hosts_in_order)),
        os=tuple((h.os for h in hosts_in_order)),
        services=tuple((tuple(h.services) for h in hosts_in_order)),
        vulnerabilities=tuple((tuple(h.vulnerabilities) for h in hosts_in_order)),
        cached_credentials=tuple((tuple(h.cached_credentials) for h in hosts_in_order)),
        system_tokens=tuple((tuple(h.system_tokens) for h in hosts_in_order)),
    )
    return EnvState(
        hosts=hosts,
        meta=meta,
        agent_ids=tuple(agent_ids),
        agent_energy=np.array(
            [int(legacy.agent_energy.get(a, 0)) for a in agent_ids], dtype=np.int32
        ),
        agent_funds=np.array(
            [int(legacy.agent_funds.get(a, 0)) for a in agent_ids], dtype=np.int32
        ),
        agent_compute=np.array(
            [int(legacy.agent_compute.get(a, 0)) for a in agent_ids], dtype=np.int32
        ),
        agent_locked_until=np.array(
            [int(legacy.agent_locked_until.get(a, 0)) for a in agent_ids],
            dtype=np.int32,
        ),
        current_tick=int(legacy.current_tick),
        business_downtime_score=float(legacy.business_downtime_score),
        knowledge=tuple(
            (frozenset(legacy.agent_knowledge.get(a, set())) for a in agent_ids)
        ),
        inventory=tuple(
            (frozenset(legacy.agent_inventory.get(a, set())) for a in agent_ids)
        ),
    )


def to_global_state(snap: EnvState):
    """Inverse of from_global_state. Discards action_history / SIEM buffer fields."""
    from netforge_rl.core.state import GlobalNetworkState, Subnet, Host

    legacy = GlobalNetworkState()
    seen = {}
    for cidr in snap.meta.subnet_cidr:
        if cidr in seen:
            continue
        sn = Subnet(cidr=cidr, name=cidr)
        seen[cidr] = sn
        legacy.add_subnet(sn)
    for i, ip in enumerate(snap.meta.ip):
        host = Host(
            ip=ip, hostname=snap.meta.hostname[i], subnet_cidr=snap.meta.subnet_cidr[i]
        )
        host.status = _decode(snap.hosts.status[i], STATUS_CODES)
        host.privilege = _decode(snap.hosts.privilege[i], PRIVILEGE_CODES)
        host.decoy = _decode(snap.hosts.decoy[i], DECOY_CODES)
        host.edr_active = bool(snap.hosts.edr_active[i])
        host.is_domain_controller = bool(snap.hosts.is_domain_controller[i])
        host.contains_honeytokens = bool(snap.hosts.contains_honeytokens[i])
        host.human_vulnerability_score = float(snap.hosts.human_vulnerability[i])
        host.cvss_score = float(snap.hosts.cvss_score[i])
        host.os = snap.meta.os[i]
        host.services = list(snap.meta.services[i])
        host.vulnerabilities = list(snap.meta.vulnerabilities[i])
        host.cached_credentials = list(snap.meta.cached_credentials[i])
        host.system_tokens = list(snap.meta.system_tokens[i])
        cid = int(snap.hosts.compromised_by_id[i])
        host.compromised_by = snap.agent_ids[cid] if cid >= 0 else 'None'
        host.system_integrity = _decode(snap.hosts.system_integrity[i], INTEGRITY_CODES)
        legacy.register_host(host)
    for j, agent in enumerate(snap.agent_ids):
        legacy.agent_energy[agent] = int(snap.agent_energy[j])
        legacy.agent_funds[agent] = int(snap.agent_funds[j])
        legacy.agent_compute[agent] = int(snap.agent_compute[j])
        legacy.agent_locked_until[agent] = int(snap.agent_locked_until[j])
        legacy.agent_knowledge[agent] = set(snap.knowledge[j])
        legacy.agent_inventory[agent] = set(snap.inventory[j])
    legacy.current_tick = int(snap.current_tick)
    legacy.business_downtime_score = float(snap.business_downtime_score)
    return legacy


_ARRAY_FIELD = {
    'status': ('status', lambda v: _encode(str(v), STATUS_CODES)),
    'privilege': ('privilege', lambda v: _encode(str(v), PRIVILEGE_CODES)),
    'decoy': ('decoy', lambda v: _encode(str(v), DECOY_CODES)),
    'edr_active': ('edr_active', bool),
    'is_domain_controller': ('is_domain_controller', bool),
    'contains_honeytokens': ('contains_honeytokens', bool),
    'human_vulnerability_score': ('human_vulnerability', float),
    'cvss_score': ('cvss_score', float),
}
_META_FIELD = {
    'os': 'os',
    'services': 'services',
    'vulnerabilities': 'vulnerabilities',
    'cached_credentials': 'cached_credentials',
    'system_tokens': 'system_tokens',
}


def _set_host_array(state, idx, field_name, encoded_value):
    arr = getattr(state.hosts, field_name).copy()
    arr[idx] = encoded_value
    new_hosts = replace(state.hosts, **{field_name: arr})
    return replace(state, hosts=new_hosts)


def _set_host_meta(state, idx, field_name, value):
    current = list(getattr(state.meta, field_name))
    if field_name in (
        'services',
        'vulnerabilities',
        'cached_credentials',
        'system_tokens',
    ):
        current[idx] = tuple(value) if not isinstance(value, str) else (value,)
    else:
        current[idx] = value
    new_meta = replace(state.meta, **{field_name: tuple(current)})
    return replace(state, meta=new_meta)


def _set_compromised_by(state, idx, agent_id):
    if agent_id == 'None' or agent_id is None:
        code = -1
    elif agent_id in state.agent_ids:
        code = state.agent_ids.index(agent_id)
    else:
        code = -1
    arr = state.hosts.compromised_by_id.copy()
    arr[idx] = code
    new_hosts = replace(state.hosts, compromised_by_id=arr)
    return replace(state, hosts=new_hosts)


def apply_state_delta(state, delta_key, delta_value=None):
    """Pure interpreter for legacy ``state_deltas`` entries."""
    if not isinstance(delta_key, str):
        return state
    parts = delta_key.split('/')
    if parts[0] == 'hosts' and len(parts) == 3:
        ip, attribute = (parts[1], parts[2])
        if ip not in state.meta.ip:
            return state
        idx = state.meta.ip.index(ip)
        if attribute == 'compromised_by':
            return _set_compromised_by(state, idx, delta_value)
        if attribute in _ARRAY_FIELD:
            field_name, encoder = _ARRAY_FIELD[attribute]
            return _set_host_array(state, idx, field_name, encoder(delta_value))
        if attribute in _META_FIELD:
            return _set_host_meta(state, idx, _META_FIELD[attribute], delta_value)
        return state
    if parts[0] == 'knowledge' and len(parts) == 3:
        agent_id, ip = (parts[1], parts[2])
        if agent_id not in state.agent_ids:
            return state
        j = state.agent_ids.index(agent_id)
        new_set = state.knowledge[j] | {ip}
        new_knowledge = tuple(
            (new_set if k == j else s for k, s in enumerate(state.knowledge))
        )
        return replace(state, knowledge=new_knowledge)
    return state


def apply_state_deltas(state, deltas):
    """Apply a dict-of-deltas or list-of-command-deltas left to right."""
    if isinstance(deltas, dict):
        for k, v in deltas.items():
            state = apply_state_delta(state, k, v)
    elif isinstance(deltas, (list, tuple)):
        for item in deltas:
            state = apply_state_delta(state, item)
    return state


def _extract_targeted_ips(state_deltas):
    """Pull every ``hosts/<ip>/...`` IP out of a state_deltas payload."""
    ips = set()
    if isinstance(state_deltas, dict):
        for key in state_deltas.keys():
            if isinstance(key, str) and key.startswith('hosts/'):
                parts = key.split('/')
                if len(parts) >= 2:
                    ips.add(parts[1])
    elif isinstance(state_deltas, (list, tuple)):
        for delta_obj in state_deltas:
            target = getattr(delta_obj, 'target_ip', None)
            if target:
                ips.add(target)
    return ips


def resolve_conflicts(effects):
    """Pure variant of ConflictResolutionEngine.resolve — does NOT mutate input."""
    from netforge_rl.core.action import ActionEffect

    blue_defended = set()
    for agent_id, eff in effects.items():
        if eff is None or not eff.success:
            continue
        if 'blue' in agent_id.lower():
            blue_defended |= _extract_targeted_ips(eff.state_deltas)
    resolved = {}
    for agent_id, eff in effects.items():
        if eff is None or not eff.success or 'red' not in agent_id.lower():
            resolved[agent_id] = eff
            continue
        red_targets = _extract_targeted_ips(eff.state_deltas)
        if red_targets & blue_defended:
            empty_deltas = [] if isinstance(eff.state_deltas, list) else {}
            new_obs = dict(eff.observation_data)
            new_obs['alert'] = 'TEMPORAL_COLLISION_DEFENSE_SUPREMACY'
            resolved[agent_id] = ActionEffect(
                success=False,
                state_deltas=empty_deltas,
                observation_data=new_obs,
                eta=eff.eta,
                action=eff.action,
            )
        else:
            resolved[agent_id] = eff
    return resolved
