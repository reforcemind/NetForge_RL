# NLP and SIEM Log Processing

Defenders in NetForge observe network activity through a simulated Security Information and Event Management (SIEM) log stream. The telemetry pipeline converts discrete state changes into realistic textual logs and vector embeddings.

## Log Generation

Whenever an agent performs an action that triggers network or system telemetry, the `SIEMLogger` formats the event using standard Windows Security Event and Sysmon schemas:

- **Process Creation (Event ID 4688 / Sysmon 1)**: `Image=powershell.exe CommandLine="Invoke-Mimikatz" ParentImage=cmd.exe`
- **Network Connection (Sysmon 3)**: `Protocol=tcp SourceIp=10.0.1.15 DestinationIp=10.0.2.8 DestinationPort=445`
- **Authentication (Event ID 4624 / 4625)**: `TargetUserName=Administrator LogonType=3 Status=0x0`

Events are queued in `siem_log_buffer` and subjected to configurable latency and benign background noise.

## Log Vectorization

To allow standard reinforcement learning policies to consume unstructured log data, `LogEncoder` provides vectorization strategies:

- **TF-IDF Encoding**: Converts the sliding window of recent log strings into fixed-length frequency vectors based on security keywords (e.g., `mimikatz`, `cve`, `failed`, `denied`).
- **Dense Transformer Embeddings**: Uses compact embedding models (via `sentence-transformers`) to generate semantic embeddings of recent incident activity.

## Language Model Interfaces

For evaluations involving Large Language Models (LLMs) acting as security analysts, `state_to_text` synthesizes the recent log buffer into readable markdown summaries, allowing LLM agents to inspect chronological activity, correlate indicators of compromise (IOCs), and select remediation actions.
