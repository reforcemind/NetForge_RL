import random
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from netforge_rl.core.state import GlobalNetworkState

N_PACKETS = 32
PACKET_DIM = 20
NODE_DIM = 8

_PROTO_ENC = {'TCP': 0.2, 'UDP': 0.4, 'ICMP': 0.6, 'Modbus': 0.8, 'S7Comm': 1.0}
_SUBNET_ENC = {'DMZ': 0.2, 'Corporate': 0.4, 'Secure': 0.6, 'OT_Subnet': 0.8}
_PRIV_ENC = {'None': 0.0, 'User': 0.5, 'Root': 1.0}
_COMMON_PORTS = [22, 80, 443, 445, 3389, 502, 102]


class PcapSynthesizer:
    """Synthesizes PCAP-like packet observations from GlobalNetworkState each tick.
    """

    def __init__(self):
        self._rng = random.Random()

    def reset(self, seed=None):
        self._rng = random.Random(seed)

    def synthesize(
        self, global_state: 'GlobalNetworkState', tick: int, max_ticks: int
    ) -> np.ndarray:
        """Returns (N_PACKETS, PACKET_DIM) float32 for the current tick."""
        sorted_ips = sorted(global_state.all_hosts.keys())
        ip_to_idx = {ip: i for i, ip in enumerate(sorted_ips)}
        tick_norm = tick / max(max_ticks, 1)
        packets = []

        c2_gateway = next(
            (h.ip for h in global_state.all_hosts.values()
             if global_state.get_subnet_name(h.subnet_cidr) == 'DMZ'
             and h.status == 'online'),
            None,
        )
        for ip, host in global_state.all_hosts.items():
            if len(packets) >= N_PACKETS - 6:
                break
            if host.compromised_by == 'None':
                continue
            dst_ip = c2_gateway or ip
            packets.append(self._packet(
                ip, dst_ip, global_state, ip_to_idx, tick_norm,
                proto='TCP', port=443,
                payload_kb=self._rng.uniform(0.1, 2.0),
                is_c2=True,
            ))

        # Hosts with elevated privilege generate lateral movement / exploit packets
        for ip, host in global_state.all_hosts.items():
            if len(packets) >= N_PACKETS - 8:
                break
            if host.privilege not in ('User', 'Root'):
                continue
            live = [h for h in global_state.all_hosts.values()
                    if h.status == 'online' and h.ip != ip]
            if not live:
                continue
            dst = self._rng.choice(live)
            proto = self._rng.choice(['TCP', 'ICMP'])
            port = self._rng.choice(_COMMON_PORTS)
            packets.append(self._packet(
                ip, dst.ip, global_state, ip_to_idx, tick_norm,
                proto=proto, port=port,
                payload_kb=self._rng.uniform(0.5, 50.0) if proto == 'TCP' else 0.08,
                is_recon=(proto == 'ICMP'),
                is_exploit=(proto == 'TCP' and port in (445, 3389)),
            ))

        # Fill remaining with benign background traffic
        live_hosts = [h for h in global_state.all_hosts.values()
                      if h.status == 'online' and '169.254' not in h.ip]
        while len(packets) < N_PACKETS and len(live_hosts) >= 2:
            src, dst = self._rng.sample(live_hosts, 2)
            packets.append(self._packet(
                src.ip, dst.ip, global_state, ip_to_idx, tick_norm,
                proto=self._rng.choice(['TCP', 'UDP']),
                port=self._rng.choice([80, 443, 22]),
                payload_kb=self._rng.uniform(0.1, 10.0),
            ))

        out = np.zeros((N_PACKETS, PACKET_DIM), dtype=np.float32)
        for i, pkt in enumerate(packets[:N_PACKETS]):
            out[i] = pkt
        return out

    def node_features(self, global_state: 'GlobalNetworkState') -> np.ndarray:
        """Returns (100, NODE_DIM) float32 host attribute matrix for GNN models.

        Columns: privilege, is_online, is_compromised, is_decoy, is_dc,
                 subnet_type, cvss_norm, edr_active.
        """
        sorted_ips = sorted(global_state.all_hosts.keys())
        feat = np.zeros((100, NODE_DIM), dtype=np.float32)
        for i, ip in enumerate(sorted_ips[:100]):
            h = global_state.all_hosts[ip]
            subnet = global_state.get_subnet_name(h.subnet_cidr)
            feat[i, 0] = _PRIV_ENC.get(h.privilege, 0.0)
            feat[i, 1] = 1.0 if h.status == 'online' else 0.0
            feat[i, 2] = 1.0 if h.compromised_by != 'None' else 0.0
            feat[i, 3] = 1.0 if h.decoy != 'inactive' else 0.0
            feat[i, 4] = 1.0 if getattr(h, 'is_domain_controller', False) else 0.0
            feat[i, 5] = _SUBNET_ENC.get(subnet, 0.0)
            feat[i, 6] = min(getattr(h, 'cvss_score', 0.0) / 10.0, 1.0)
            feat[i, 7] = 1.0 if getattr(h, 'edr_active', False) else 0.0
        return feat

    def _packet(
        self,
        src_ip: str,
        dst_ip: str,
        global_state: 'GlobalNetworkState',
        ip_to_idx: dict,
        tick_norm: float,
        proto: str = 'TCP',
        port: int = 80,
        payload_kb: float = 1.0,
        is_c2: bool = False,
        is_recon: bool = False,
        is_exploit: bool = False,
    ) -> np.ndarray:
        sh = global_state.all_hosts.get(src_ip)
        dh = global_state.all_hosts.get(dst_ip)
        src_sn = global_state.get_subnet_name(sh.subnet_cidr) if sh else ''
        dst_sn = global_state.get_subnet_name(dh.subnet_cidr) if dh else ''

        pkt = np.zeros(PACKET_DIM, dtype=np.float32)
        pkt[0] = ip_to_idx.get(src_ip, 0) / 100.0
        pkt[1] = ip_to_idx.get(dst_ip, 0) / 100.0
        pkt[2] = _PROTO_ENC.get(proto, 0.2)
        pkt[3] = min(port, 65535) / 65535.0
        pkt[4] = min(payload_kb / 100.0, 1.0)
        pkt[5] = 1.0 if (proto == 'TCP' and not is_c2) else 0.0   # SYN
        pkt[6] = 0.0                                                # RST
        pkt[7] = 1.0 if is_c2 else 0.0                             # ACK
        pkt[8] = 1.0 if payload_kb > 0.5 else 0.0                  # PSH
        pkt[9] = 1.0 if src_sn != dst_sn else 0.0                  # lateral
        pkt[10] = float(is_c2)
        pkt[11] = float(is_recon)
        pkt[12] = 1.0 if (is_c2 and payload_kb > 5.0) else 0.0    # exfil
        pkt[13] = float(is_exploit)
        pkt[14] = 1.0 if dst_sn in ('Secure', 'OT_Subnet') else 0.0
        pkt[15] = _PRIV_ENC.get(getattr(sh, 'privilege', 'None'), 0.0)
        pkt[16] = 1.0 if (dh and dh.compromised_by != 'None') else 0.0
        pkt[17] = tick_norm
        pkt[18] = 1.0 if (proto == 'TCP' and port in (443, 22)) else 0.0  # encrypted
        pkt[19] = 1.0 if (is_c2 or is_exploit) else (0.5 if is_recon else 0.0)
        return pkt
