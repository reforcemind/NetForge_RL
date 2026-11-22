# Zero-Trust Identity Architecture

Traditional network defense models rely primarily on network perimeter boundaries. In modern zero-trust architectures, authentication and authorization are continuously validated at every transaction.

NetForge implements explicit identity mechanics modeling enterprise credential lifecycles and ticket-granting services.

## Token Lifecycle

Each endpoint maintains an internal token array representing cached session credentials:

- **Initial Issuance**: When users authenticate to workstations, identity tokens (Kerberos tickets, NTLM hashes) are generated and stored in local memory.
- **Credential Harvesting**: If an adversary obtains `Root` privileges on a host, executing `DumpLSASS` extracts valid tokens from memory.
- **Identity-Based Lateral Movement**: With extracted tokens, the adversary can execute `PassTheHash` or `PassTheTicket` to access remote endpoints without needing unpatched network vulnerabilities or remote service exploits.

## Defensive Countermeasures

Blue defenders can disrupt credential-driven attacks using identity-aware actions:

- **`RotateKerberos`**: Executing key rotation on the Domain Controller invalidates all existing Kerberos ticket-granting tickets across the enterprise.
- **Stale Token Detection**: When an adversary attempts lateral movement using an invalidated token, the authentication handshake fails immediately and triggers a high-severity authentication alert (Event ID 4625) in the SIEM buffer.
- **Micro-Segmentation**: Coupled with `ConfigureACL`, zero-trust policies restrict lateral communication paths, isolating compromised identity tiers before privileges spread across domain boundaries.
