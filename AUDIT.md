# NetForge_RL — Phase 0 Audit

**Branch:** `chore/audit-baseline`
**Date:** 2026-05-30
**Auditor:** automated harness + manual review
**Target:** NeurIPS Datasets & Benchmarks 2026 submission

---

## 1. Repository inventory

| Area | Path | Status |
|------|------|--------|
| Core physics | [netforge_rl/core/](netforge_rl/core/) | ✅ clean module separation (`state`, `action`, `observation`, `physics`, `registry`) |
| MARL env (PettingZoo) | [netforge_rl/environment/parallel_env.py](netforge_rl/environment/parallel_env.py) | ✅ correct `(obs, rewards, term, trunc, infos)` API, ❌ stateful & not vmap-able |
| Actions | [netforge_rl/actions/](netforge_rl/actions/) (red/blue/network) | ✅ 32 actions registered via `action_registry` |
| Scenarios | [netforge_rl/scenarios/](netforge_rl/scenarios/) | ✅ ransomware, APT espionage |
| Topologies | [netforge_rl/topologies/network_generator.py](netforge_rl/topologies/network_generator.py) | ✅ procedural 3-tier generator |
| SIEM / NLP | [netforge_rl/siem/](netforge_rl/siem/), [netforge_rl/nlp/](netforge_rl/nlp/) | ✅ TF-IDF encoder, Windows event templates |
| Sim2Real | [netforge_rl/sim2real/](netforge_rl/sim2real/) | ✅ Mock + Docker hypervisors behind a single flag |
| Tests | [tests/](tests/) | ✅ 79 passing |
| Render | `parallel_env.render()` | ❌ no-op |
| Baselines | — | ❌ none committed |
| Notebooks | — | ❌ none |
| JAX backend | — | ❌ does not exist |

---

## 2. Measured baseline (legacy PyTorch backend)

Harness: [benchmarks/sps_baseline.py](benchmarks/sps_baseline.py) — raw result archived at [benchmarks/results/sps_baseline.json](benchmarks/results/sps_baseline.json).

| Metric | Value |
|---|---|
| Backend | `legacy-pytorch` (PettingZoo ParallelEnv, single instance) |
| Scenario | `ransomware` |
| Episodes | 3 |
| Max ticks | 200 |
| **SPS — mean** | **~10.4** |
| SPS — median | ~10.7 |
| SPS — min / max | 10.4 / 10.8 |
| Python | 3.12.10 |
| Platform | Windows 11 |

**Interpretation:** at ~10 SPS per instance, a 1M-step PPO training run takes ≈28 hours of wall-clock on a single env. This is the throughput crisis the JAX backend (Phase 2) is designed to resolve. Target: ≥10⁵ SPS aggregate at 4096 envs on an A100 — a ~10⁴× improvement.

---

## 3. Golden trajectory lock

Test: [tests/parity/test_golden_trajectory.py](tests/parity/test_golden_trajectory.py)
Helper: [tests/parity/trajectory_fingerprint.py](tests/parity/trajectory_fingerprint.py)

```
GOLDEN_FINGERPRINT = "abd164a5ef403918dced139e96010322625ac3a31e3af46fd7d5ad7b669912a2"
seed=42, max_ticks=25, scenario=ransomware
```

The fingerprint hashes the quantized (`6 dp`) reward stream + per-step termination/truncation masks. Determinism verified across re-rolls. **Every later refactor PR must keep this test green** (or update the hash with a `CHANGELOG.md` entry explaining the intentional drift).

We deliberately do **not** hash full observations: the SIEM TF-IDF features depend on Python hash-seed randomization, which would force `PYTHONHASHSEED=0` on every consumer. Rewards + termination capture the macro-dynamics that matter for RL parity.

---

## 4. Refactor blockers (must be resolved before Phase 2 JAX backend)

| # | Blocker | Location | Phase |
|---|---|---|---|
| B1 | `dict[ip → HostObject]` state is not a PyTree — cannot `jax.vmap` | [core/state.py:5](netforge_rl/core/state.py#L5) | Phase 1 |
| B2 | In-place mutation via `apply_delta` and `event_queue.append` | [environment/parallel_env.py:250](netforge_rl/environment/parallel_env.py#L250) | Phase 1 |
| B3 | Implicit RNG (`np.random` inside scenarios/generator) — JAX needs explicit keys | [topologies/network_generator.py](netforge_rl/topologies/network_generator.py) | Phase 1 |
| B4 | Hardcoded shape constants leak through API (256 obs, 100 hosts, 132 mask) | [environment/parallel_env.py:91-101](netforge_rl/environment/parallel_env.py#L91-L101) | Phase 1 |
| B5 | `render()` is a no-op — no visualization for paper/leaderboard | [environment/parallel_env.py:449](netforge_rl/environment/parallel_env.py#L449) | Phase 4 |
| B6 | No baseline algorithms committed, no leaderboard | — | Phase 5 |
| B7 | No Colab/Jupyter onboarding | — | Phase 6 |
| B8 | Docs README links point to `xaiqo/NetForge_RL`; project lives at `reforcemind/NetForge_RL` | [README.md:11](README.md#L11) | Phase 6 |

---

## 5. What we keep, untouched

These modules are domain-correct and competitive moats — we wrap, never rewrite:

- MITRE ATT&CK action taxonomy ([actions/](netforge_rl/actions/))
- Zero-Trust Identity token enforcement (in `core/state.py` host fields)
- Sim2Real Docker bridge ([sim2real/bridge.py](netforge_rl/sim2real/bridge.py))
- SIEM event templates ([siem/event_templates.py](netforge_rl/siem/event_templates.py))
- Honeytoken / decoy logic ([environment/parallel_env.py:365-377](netforge_rl/environment/parallel_env.py#L365-L377))

---

## 6. CI / quality gates introduced

A PR may merge into `main` only if:

1. `pytest tests/` — all green (currently 79 + 2 new = **81 tests**).
2. `pytest tests/parity/test_golden_trajectory.py` — golden hash matches *or* PR updates `GOLDEN_FINGERPRINT` with a justification in `CHANGELOG.md`.
3. `python -m benchmarks.sps_baseline` — SPS regression ≤ 5 % relative to the previous tagged release (gate enforced from Phase 1 onward).
4. Squash-merge with linear history.

---

## 7. Exit criteria for Phase 0 — ✅ met

- [x] Existing test suite verified green (79/79).
- [x] SPS baseline harness committed and producing a JSON record.
- [x] Golden trajectory fingerprint locked behind a regression test.
- [x] Audit document written with measured numbers and named blockers.
- [x] Branching strategy in place (`chore/audit-baseline` → `main` after review).

Next: open `refactor/functional-core` (Phase 1) — extract `EnvState` as a frozen PyTree, make `step_state` pure, route all RNG through explicit keys. Acceptance: golden fingerprint preserved; SPS within ±10 % of baseline.
