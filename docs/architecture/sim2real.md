# Sim2Real Bridge

NetForge provides a containerized execution bridge (`netforge_rl/docker_bridge/`) allowing abstract simulation actions to trigger real commands inside live container environments (e.g., Docker, Vulhub).

## Architectural Flow

```
┌────────────────────┐
│ Simulation Action  │  (e.g., ExploitEternalBlue on host 5)
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   Docker Bridge    │  Maps action to real exploit script / CLI command
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Docker / Container │  Executes command against live vulnerable service
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Output Parser      │  Matches stdout/stderr against success signatures
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ State Synchronization Update GlobalNetworkState (e.g., privilege -> Root)
└────────────────────┘
```

## How It Works

1. **Action Translation**: When the bridge is active, discrete actions map to command templates. For instance, `ExploitEternalBlue` triggers a Metasploit or Python script targeting an emulated SMB daemon inside a Docker container.
2. **Execution & Signature Verification**: The command runs inside an isolated bridge network. The container's exit code, stdout, and stderr are evaluated against deterministic regex signatures (e.g., finding successful shell prompts or meterpreter sessions).
3. **State Reflection**: On a confirmed exploit, the simulation's `GlobalNetworkState` is updated to reflect the new privilege level, synchronizing abstract RL states with physical container reality.
