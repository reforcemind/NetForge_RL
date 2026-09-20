# Zero-Trust Identity

Each node holds a token array. Identity lateral movement (`PassTheHash`,
`PassTheTicket`) checks tokens, not CVEs. Blue `RotateKerberos` clears them;
stale tokens fail and log to SIEM.
