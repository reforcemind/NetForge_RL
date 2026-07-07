RED_COMPROMISE = 0
RED_PRIVESC = 1
RED_IMPACT = 2
RED_KINETIC = 3
RED_EXPLOIT_BLUEKEEP = 4
RED_EXPLOIT_ETERNALBLUE = 5
RED_EXPLOIT_HTTP_RFI = 6
RED_RECON = 7
RED_EXFILTRATE = 8  # Root-gated; +exfiltrated_bytes per tick
RED_SHARE_INTEL = 9  # OR every Red row of knowledge_mask together
RED_DUMP_LSASS = 10  # Root-gated; OR host_tokens[target] -> agent_credentials
RED_PASS_THE_HASH = 11  # compromise gated on token-locality (target's required token)
RED_PASS_THE_TICKET = 12  # privesc gated on token-locality
RED_JUICY_POTATO = 13  # Windows-only privesc (os_family == WINDOWS)
RED_V4L2 = 14  # Linux-only privesc (os_family == LINUX)
RED_KILL_PROCESS = 15  # Root-gated; status -> kernel_panic
RED_IP_FRAGMENTATION = 16  # acts as generic compromise in JAX
RED_NETWORK_SCAN = 17  # recon action, adds to knowledge
RED_DISCOVER_REMOTE_SYSTEMS = 18  # recon action, adds to knowledge
RED_SPEARPHISHING = 19  # compromise gated by OS_WINDOWS

BLUE_ISOLATE = 0
BLUE_RESTORE = 1
BLUE_DEPLOY_DECOY = 2
BLUE_DEPLOY_HONEYTOKEN = 3
BLUE_REMOVE = 4
BLUE_SAT = 5
BLUE_MONITOR = 6
BLUE_MISINFORM_APACHE = 7  # decoy -> Apache (planted fake service)
BLUE_CONFIGURE_ACL = 8  # edr_active -> True (endpoint monitoring on)
BLUE_ROTATE_KERBEROS = 9  # clear every Red row of agent_credentials
BLUE_ANALYZE = 10  # acts identically to monitor but targeted
BLUE_RESTORE_FROM_BACKUP = 11  # also clears system_integrity
BLUE_MISINFORM_TOMCAT = 12  # decoy -> Tomcat
BLUE_MISINFORM_SSHD = 13  # decoy -> SSHD

N_RED_ACTIONS = 20
N_BLUE_ACTIONS = 14

SAT_DROP = 0.1  # human-vulnerability reduction per security-awareness-training action
EXFIL_PER_HOST = 5.0  # bytes-units per Rooted host per Exfiltrate tick
