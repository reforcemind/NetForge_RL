# Action Coverage Audit — JAX vs Legacy Backend

**Status at HEAD (`ad6e9dc`).** Sourced by introspecting `action_registry`
plus reading `netforge_rl/backends/jax/vector_env.py`.

The repo has two backends sharing one functional core ([core/functional.py](../netforge_rl/core/functional.py)).
The **legacy PyTorch / PettingZoo backend** ([environment/parallel_env.py](../netforge_rl/environment/parallel_env.py))
wires up all 34 actions registered in `action_registry`. The **JAX backend**
([backends/jax/vector_env.py](../netforge_rl/backends/jax/vector_env.py)) is a
parallel reimplementation optimized for `jax.vmap` — it currently implements
8 of those 34 action semantics.

Concretely: training via the JAX path runs at ~660k SPS on 4096 envs but
the policy's action vocabulary is restricted to the 8 ported actions.
Training via the legacy path covers the full 34 but at ~10 SPS.

---

## 1. Registry inventory

34 action classes across 6 team buckets (gid = action_type integer 0–31 in
the MultiDiscrete action space). When an agent_id resolves through
`get_action_class`, the primary team is checked first, then the agent-
specific team — so e.g. red_operator's gid=2 ("Impact") is shadowed by
red's gid=2 ("DiscoverNetworkServices") and is unreachable via that path.

| Team | Actions |
|---|---|
| `red` (8) | ExploitRemoteService, PrivilegeEscalate, DiscoverNetworkServices, ExploitBlueKeep, ExploitEternalBlue, ExploitHTTP_RFI, DumpLSASS, PassTheTicket |
| `red_commander` (4) | NetworkScan, DiscoverRemoteSystems, DiscoverNetworkServices, ShareIntelligence |
| `red_operator` (8) | Impact, JuicyPotato, V4L2KernelExploit, KillProcess, PassTheHash, ExfiltrateData, OverloadPLC, SpearPhishing |
| `blue` (6) | IsolateHost, RestoreHost, Remove, RestoreFromBackup, ConfigureACL, SecurityAwarenessTraining |
| `blue_commander` (6) | RotateKerberos, DecoyApache, DecoySSHD, DecoyTomcat, Misinform, DeployHoneytoken |
| `blue_operator` (2) | Monitor, Analyze |

---

## 2. JAX backend coverage

| JAX action | Legacy equivalent | Mutates | Reward | Shipped |
|---|---|---|---|---|
| `RED_COMPROMISE` (0) | ExploitRemoteService + variants | `privilege: None→User`, `compromised_by` | +1 | ✅ |
| `RED_PRIVESC` (1) | PrivilegeEscalate / JuicyPotato | `privilege: User→Root` (gated) | +3 | ✅ |
| `RED_IMPACT` (2) | Impact | `system_integrity → compromised` (Root-gated) | +10 | ✅ |
| `RED_KINETIC` (3) | OverloadPLC | `system_integrity → kinetic_destruction` (Root-gated) | +10,000 | ✅ |
| `BLUE_ISOLATE` (0) | IsolateHost | `status → isolated` | +1 | ✅ |
| `BLUE_RESTORE` (1) | RestoreHost / RestoreFromBackup | `priv → None`, `status → online`, `integrity → clean`, `compromised_by → -1` | +2 | ✅ |
| `BLUE_DEPLOY_DECOY` (2) | DecoyApache / SSHD / Tomcat | `decoy → active` | +0.5 | ✅ |
| `BLUE_DEPLOY_HONEYTOKEN` (3) | DeployHoneytoken | `contains_honeytokens → True`; Red trap penalty -5 | +0.5 | ✅ |
| `BLUE_REMOVE` (4) | Remove | `privilege → None` (status unchanged) | +1.5 | ✅ |
| `BLUE_SAT` (5) | SecurityAwarenessTraining | `human_vulnerability -= 0.1` (clamped at 0) | +0.3 | ✅ |
| `RED_EXPLOIT_BLUEKEEP` (4) | ExploitBlueKeep | COMPROMISE gated on `vuln_mask[CVE-2019-0708]` | +1.5 | ✅ |
| `RED_EXPLOIT_ETERNALBLUE` (5) | ExploitEternalBlue | COMPROMISE gated on `vuln_mask[MS17-010]` | +1.5 | ✅ |
| `RED_EXPLOIT_HTTP_RFI` (6) | ExploitHTTP_RFI | COMPROMISE gated on `vuln_mask[CVE-2021-44228]` | +1.5 | ✅ |
| `RED_RECON` (7) | DiscoverRemoteSystems / NetworkScan | `knowledge_mask[red_agent, target] = True` | +0.2 on new intel | ✅ |
| `BLUE_MONITOR` (6) | Monitor / Analyze | `knowledge_mask[blue_agent, target] = True` | +0.2 on new intel | ✅ |
| `RED_EXFILTRATE` (8) | ExfiltrateData | `exfiltrated_bytes += EXFIL_PER_HOST` per Rooted target | +EXFIL_PER_HOST | ✅ |
| `BLUE_MISINFORM` (7) | Misinform | `decoy → Apache` (planted fake service) | +0.4 | ✅ |

**Coverage:** 9 red + 8 blue = 17 unique behaviours. Conflict resolution
between any same-target Red/Blue pair holds via `resolve_conflicts_mask`.

---

## 3. What's missing — by required scaffolding

The remaining 26 ports fall into four buckets keyed by what new state-array
fields they require. Easier buckets first.

### 3.1 Already representable — drop-in adds (~0 new fields)

Should land as straightforward kernel additions matching the existing
COMPROMISE / RESTORE pattern. No JaxEnvState changes.

| Legacy action | What it does | Cost |
|---|---|---|
| ExploitBlueKeep, ExploitEternalBlue, ExploitHTTP_RFI | CVE-gated variants of compromise | per host: condition on `vulnerabilities[idx]` (already in `meta`, not yet a leaf — push to a `bool[N_HOSTS, N_CVE]` mask). |
| JuicyPotato, V4L2KernelExploit | OS-gated privesc | condition on `os` (already in meta). |
| Remove | `privilege → None` without restoring status | trivial — one-line variant of BLUE_RESTORE. |
| RestoreFromBackup | Same as BLUE_RESTORE plus integrity wipe | already what BLUE_RESTORE does. |
| KillProcess | small impact / disrupt | new `process_running` bool field. |
| Misinform | Blue commander; flips one host's `decoy` to a specific category | trivial variant of BLUE_DEPLOY_DECOY. |
| SecurityAwarenessTraining | bumps `human_vulnerability_score` down | trivial — additive arithmetic on the existing float32 field. |

### 3.2 Need a new fog-of-war / knowledge mask

A `knowledge: bool[N_AGENTS, N_HOSTS]` field on `JaxHostArrays`, plus a
helper that sets bits on reconnaissance success.

| Legacy action | Behaviour |
|---|---|
| DiscoverNetworkServices, DiscoverRemoteSystems, NetworkScan | flips `knowledge[agent, target]` (and neighbors). |
| Monitor, Analyze | Blue equivalents — reveals host status / privilege. |
| ShareIntelligence | OR's knowledge rows across allied agents. |

Cost: adds one rank-2 leaf array (proportional to `N_AGENTS × N_HOSTS = 4 × 100`)
and an update kernel. Converters need to round-trip it.

### 3.3 Need an identity / credential mask

A `credentials: bool[N_AGENTS, N_TOKEN]` field. The legacy state has
`agent_inventory: dict[agent_id, set[str]]` — we'd pin the token vocabulary
to a small static list (e.g. `('Enterprise_Admin', 'Local_Admin_DMZ', ...)`).

| Legacy action | Behaviour |
|---|---|
| DumpLSASS | Reads `cached_credentials` off the host → adds to Red's inventory. |
| PassTheTicket, PassTheHash | Spends an inventory token to unlock a lateral move. |
| RotateKerberos | Blue commander; clears every Red inventory bit for that token. |

Cost: one rank-2 leaf, one static token codebook.

### 3.4 Need new dynamics

| Legacy action | Why it's harder |
|---|---|
| ExfiltrateData | Per-step cumulative damage — needs a `bytes_exfiltrated: float32` scalar per env. Cheap. |
| SpearPhishing | Probabilistic; needs explicit `jax.random.PRNGKey` threading in the step (currently absent — the step is deterministic). Bigger lift. |
| ConfigureACL | Touches the (currently legacy-only) firewall rules. Need a firewall mask field. |

### 3.5 Won't port (out of scope for the JAX kernel)

- `Sim2RealBridge` Docker hypervisor actions — by design only run on the legacy
  backend (eval-time only).
- Anything that mutates the agent's own action_history (legacy
  `required_prior_state` machinery). Already replaced in JAX by direct
  state gating (e.g. RED_IMPACT requires Root, not "PrivilegeEscalate has
  fired on this host before").

---

## 4. Recommended porting order

Each line is roughly one PR. Tight scope, hash-preserving (legacy untouched).

1. **CVE-gated exploit variants** (BlueKeep, EternalBlue, HTTP_RFI). Adds a
   `vuln_mask: bool[N_HOSTS, N_CVE]` leaf and conditions COMPROMISE on it.
2. **OS-gated privesc** (JuicyPotato, V4L2KernelExploit). Adds an `os_code:
   int8[N_HOSTS]` leaf (already in meta, just promote).
3. **Knowledge mask + reconnaissance trio** (DiscoverRemoteSystems,
   NetworkScan, Monitor). Adds `knowledge: bool[N_AGENTS, N_HOSTS]`.
4. **Credential mask + DumpLSASS / PassTheTicket / RotateKerberos**. Adds
   `credentials: bool[N_AGENTS, N_TOKEN]`.
5. **ExfiltrateData** + cumulative damage scalar.
6. **SpearPhishing** — gated on RNG plumbing into the step (this is the
   one that motivates the `key` argument we've been deferring on the
   batched step signature).

---

## 5. Beyond actions — broader state of the system

| Subsystem | Status |
|---|---|
| Audit / Phase 0 baseline | ✅ shipped ([AUDIT.md](AUDIT.md)) |
| Phase 1 functional core | ✅ shipped |
| Phase 2 JAX kernels + vmap | ✅ shipped (8 actions) |
| Phase 3 bridges (JaxMARL / DLPack / CleanRLVec) | ✅ shipped |
| Phase 4 decoupled renderer | ✅ shipped |
| Phase 5 baselines (random / heuristic / JAX PPO) | ✅ shipped |
| Phase 5 scenarios | ✅ 4 of 5 (ransomware / apt / iot_grid / ot_stuxnet) — `cloud_hybrid` deferred |
| Phase 5 leaderboard | ✅ shipped (`leaderboard/baselines{,_summary}.json`) |
| Phase 6 notebooks 01-06 | ✅ shipped |
| Phase 7 CHANGELOG / DATASHEET | ✅ shipped (CHANGELOG has post-v4.0 commits to backfill) |
| Phase 8 M1 LA + VLA wrappers | ✅ shipped |
| Phase 8 M2 zero-shot leaderboard | ✅ shipped (real Anthropic + OpenAI client adapters) |
| Phase 8 M3 LoRA + PPO fine-tune scaffold + Colab | ✅ shipped (notebook 07) |
| **JAX backend full 32 action port** | ⚠️ 8 of 34 |
| **Phase 8 grammar-constrained decoding** | ⚠️ regex parser only — grammar/tool-use schemas under `semantic/grammars/` not yet written |
| **MkDocs nav refresh + site rebuild** | ⚠️ `mkdocs.yml` still points at the old branding |
| **Phase 5 cloud_hybrid scenario** | ⚠️ not implemented |
| **Phase 8 M2 Google client** | ⚠️ not implemented (Anthropic + OpenAI shipped) |
| **Phase 7 release tag + Zenodo DOI** | ⚠️ user-driven; tag not pushed |

---

## 6. Concrete next slices (ordered by leverage)

1. CVE-gated Red exploit variants (§3.1, §4-1).
2. OS-gated Red privesc (§3.1, §4-2).
3. Knowledge mask + reconnaissance trio (§3.2, §4-3).
4. CHANGELOG sync for post-v4.0 commits.
5. MkDocs nav refresh.
6. Phase 8 grammar-constrained decoding hooks (Anthropic tool_use schema first).
7. Credential mask + LSASS / PassTheTicket (§3.3, §4-4).
8. cloud_hybrid scenario.
9. ExfiltrateData + cumulative damage scalar (§3.4, §4-5).
10. SpearPhishing — motivates threading a `jax.random.PRNGKey` through the step (§3.4, §4-6).
