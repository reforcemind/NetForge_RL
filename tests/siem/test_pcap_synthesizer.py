import numpy as np
from netforge_rl.siem.pcap_synthesizer import (
    PcapSynthesizer,
    N_PACKETS,
    PACKET_DIM,
    NODE_DIM,
)
from netforge_rl.core.state import GlobalNetworkState, Host, Subnet


def test_pcap_synthesizer_reset():
    synth = PcapSynthesizer()
    synth.reset(42)
    val1 = synth._rng.random()
    synth.reset(42)
    val2 = synth._rng.random()
    assert val1 == val2


def test_pcap_synthesizer_node_features():
    synth = PcapSynthesizer()
    state = GlobalNetworkState()

    host1 = Host('192.168.1.1', 'h1', '192.168.1.0/24')
    host1.privilege = 'Root'
    host1.status = 'online'
    host1.compromised_by = 'red'
    host1.decoy = 'Apache'
    host1.is_domain_controller = True
    host1.edr_active = True
    host1.cvss_score = 7.5

    subnet = Subnet('192.168.1.0/24', 'DMZ')
    state.add_subnet(subnet)
    state.register_host(host1)

    feats = synth.node_features(state)
    assert feats.shape == (100, NODE_DIM)

    h1_idx = 0
    assert feats[h1_idx, 0] == 1.0
    assert feats[h1_idx, 1] == 1.0
    assert feats[h1_idx, 2] == 1.0
    assert feats[h1_idx, 3] == 1.0
    assert feats[h1_idx, 4] == 1.0
    assert feats[h1_idx, 5] == 0.2
    assert feats[h1_idx, 6] == 0.75
    assert feats[h1_idx, 7] == 1.0


def test_pcap_synthesizer_synthesize():
    synth = PcapSynthesizer()
    state = GlobalNetworkState()

    subnet = Subnet('192.168.1.0/24', 'DMZ')
    state.add_subnet(subnet)

    host1 = Host('192.168.1.1', 'h1', '192.168.1.0/24')
    host1.privilege = 'Root'
    host1.status = 'online'
    host1.compromised_by = 'red'

    host2 = Host('192.168.1.2', 'h2', '192.168.1.0/24')
    host2.privilege = 'User'
    host2.status = 'online'
    host2.compromised_by = 'None'

    host3 = Host('192.168.1.3', 'h3', '192.168.1.0/24')
    host3.privilege = 'None'
    host3.status = 'online'
    host3.compromised_by = 'None'

    state.register_host(host1)
    state.register_host(host2)
    state.register_host(host3)

    packets = synth.synthesize(state, tick=5, max_ticks=10)
    assert packets.shape == (N_PACKETS, PACKET_DIM)

    assert np.any(packets > 0.0)
