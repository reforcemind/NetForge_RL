# Zero-Trust Identity

The Zero-Trust Identity module governs authorization tokens and their alignment with `GlobalNetworkState` access vectors.

## Mechanics

- **Token Arrays**: Each active node possesses a continuous token array representing authorized privilege contexts.
- **Verification**: Red agents executing identity-based lateral movement (e.g., `PassTheHash`, `PassTheTicket`) are validated strictly against this array rather than service vulnerability flags.
- **Invalidation**: Blue agents execute actions like `RotateKerberos` to forcefully clear token arrays globally. If a Red agent attempts to utilize an invalidated token for authentication, the action `success` flag returns `False` and a failed authentication event is passed to the SIEM buffer.