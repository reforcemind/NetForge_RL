# Threat model

Actions are mechanics, not dice rolls.

| | |
|---|---|
| Spearphish | `human_vulnerability_score`; no routing |
| BlueKeep / EternalBlue / HTTP RFI | RDP / SMB / HTTP CVE flags |
| JuicyPotato / V4L2 | DCOM / Linux kernel flags |
| Pass-the-hash | DC tokens, not CVEs |
| Honeytoken | unmaskable SIEM on ingest |
| Decoys | Apache / Tomcat / SSH sinkholes |

Red visibility updates only on successful discovery. Blue visibility updates
only when an action emits a SIEM signature.
