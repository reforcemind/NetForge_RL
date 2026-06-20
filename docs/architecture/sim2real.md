# Sim2Real Bridge

The Sim2Real Bridge interfaces internal RL action tensors into executable network commands for live environments, such as Docker containers (e.g., Vulhub).

## Mechanics

- **Translation Mapping**: Maps discrete RL action integers (e.g., `ActionType.ExploitEternalBlue`) to standardized payload sequences (e.g., Metasploit RPC commands).
- **Execution Engine**: Interacts with the live target via a strictly typed execution wrapper. It captures the physical stdout/stderr output resulting from the payload injection.
- **State Feedback**: Evaluates the output against expected success signatures. If validation passes, the corresponding simulated `GlobalNetworkState` tensor is updated to reflect the physical reality (e.g., escalating privilege to `Root`).