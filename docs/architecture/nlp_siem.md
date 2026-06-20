# NLP-SIEM Pipeline

The NLP-SIEM Pipeline translates raw network telemetry strings into vectorized observation spaces or natural language prompts specifically for Blue agents.

## Mechanics

- **Event Generation**: The `ConflictResolutionEngine` processes successful and failed state modifications, emitting raw text signatures matching Sysmon/Windows Event Logs (e.g., Event ID 4624 for `Logon`).
- **Log Buffer**: These raw strings are appended to a rolling global event queue (`siem_log_buffer`), which simulates the real-time logging pipeline.
- **Vectorization (RL Agents)**: For standard neural architectures, the most recent $N$ logs in the buffer are mapped through a static TF-IDF encoder or dense embedding layer, transforming the strings into a fixed-size `BlueAgentObservationVector`.
- **Text Translation (LLM Agents)**: For foundation models, the `state_to_text` function directly pulls raw text from the buffer and injects it into a structured natural language prompt, bypassing the vectorization phase entirely.