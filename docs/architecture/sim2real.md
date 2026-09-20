# Sim2Real Bridge

Maps discrete actions (e.g. `ExploitEternalBlue`) to live commands (Docker /
Vulhub). Stdout/stderr is matched against success signatures; on hit, the
sim `GlobalNetworkState` is updated (e.g. privilege → Root).
