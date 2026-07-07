ATTACK_TECHNIQUES: dict[str, tuple[str, str, str]] = {
    'NetworkScan': ('T1046', 'Network Service Discovery', 'Discovery'),
    'DiscoverNetworkServices': ('T1046', 'Network Service Discovery', 'Discovery'),
    'DiscoverRemoteSystems': ('T1018', 'Remote System Discovery', 'Discovery'),
    'ExploitRemoteService': (
        'T1210',
        'Exploitation of Remote Services',
        'Lateral Movement',
    ),
    'ExploitBlueKeep': ('T1210', 'Exploitation of Remote Services', 'Lateral Movement'),
    'ExploitEternalBlue': (
        'T1210',
        'Exploitation of Remote Services',
        'Lateral Movement',
    ),
    'ExploitHTTP_RFI': (
        'T1190',
        'Exploit Public-Facing Application',
        'Initial Access',
    ),
    'SpearPhishing': ('T1566', 'Phishing', 'Initial Access'),
    'PrivilegeEscalate': (
        'T1068',
        'Exploitation for Privilege Escalation',
        'Privilege Escalation',
    ),
    'V4L2KernelExploit': (
        'T1068',
        'Exploitation for Privilege Escalation',
        'Privilege Escalation',
    ),
    'JuicyPotato': ('T1134', 'Access Token Manipulation', 'Privilege Escalation'),
    'DumpLSASS': ('T1003.001', 'OS Credential Dumping: LSASS Memory', 'Credential Access'),
    'PassTheHash': ('T1550.002', 'Use Alternate Authentication Material: Pass the Hash',
                    'Lateral Movement'),
    'PassTheTicket': ('T1550.003',
                      'Use Alternate Authentication Material: Pass the Ticket',
                      'Lateral Movement'),
    'ExfiltrateData': ('T1041', 'Exfiltration Over C2 Channel', 'Exfiltration'),
    'Impact': ('T1486', 'Data Encrypted for Impact', 'Impact'),
    'KillProcess': ('T1489', 'Service Stop', 'Impact'),
    'OverloadPLC': ('T0831', 'Manipulation of Control', 'ICS Impact'),
    'IPFragmentationAction': (
        'T1027',
        'Obfuscated Files or Information',
        'Defense Evasion',
    ),
}

ALL_TECHNIQUE_IDS = frozenset(t[0] for t in ATTACK_TECHNIQUES.values())

def technique_for(action_name: str):
    """Return (technique_id, technique_name, tactic) for a red action, or None."""
    return ATTACK_TECHNIQUES.get(action_name)
