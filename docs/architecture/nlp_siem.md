# NLP-SIEM

Conflict resolution emits Sysmon-like strings into `siem_log_buffer`.
RL agents encode the last N logs (TF-IDF or dense). LLM agents get
`state_to_text` from the same buffer.
