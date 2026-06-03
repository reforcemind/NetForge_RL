# NetForge RL: Comprehensive NeurIPS Evaluations & Datasets Audit

This document serves as the master audit, risk map, and strategic positioning guide for submitting NetForge RL to the **NeurIPS Evaluations & Datasets** track (2026). It evaluates the project across competitive landscape positioning, code health, artifact quality, and strategic gaps.

---

## 1. The Competitive Landscape & Core Pitch

The most critical strategic insight is how NetForge compares to the broader RL ecosystem. The landscape currently looks like this:

| Environment | Speed | Cyber-faithful | Multi-agent | LLM-native | Diagnostic |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CybORG / CAGE** | slow | ✅ | ✅ | ❌ | ❌ |
| **JaxMARL / SMAX** | fast | ❌ | ✅ | ❌ | ❌ |
| **bsuite (DM)** | fast | ❌ | ❌ | ❌ | ✅ |
| **Melting Pot (DM)** | medium | ❌ | ✅ | ❌ | ❌ |
| **Acme + dm_env** | n/a (framework) | ❌ | partial | ❌ | ❌ |
| **NetworkGym (Intel)**| medium | network-ops | ❌ | ❌ | ❌ |
| **NetForge today** | fast (JAX) + slow | ✅ | ✅ | ✅ Phase 8 | ❌ |

**The Pitch:** NetForge uniquely owns the intersection of **cyber-faithful × multi-agent × LLM-native × fast**. However, to appeal directly to the *Evaluations & Datasets* reviewers, it must not only be a "fitness benchmark" (scenarios → leaderboard), but also a **bsuite-style diagnostic suite**. 

No one has built a focused unit-test diagnostic suite for cyber. By introducing diagnostic scenarios ("Can your trained PPO remember a Red plant from tick 0 to 50?", "Can it assign credit for a kinetic reward 80 ticks later?", "Does it overexplore when SIEM noise dominates?"), NetForge delivers **two contributions in one paper**.

---

## 2. Features to Borrow (DeepMind / NetworkGym)

To fully realize the diagnostic potential and improve research ergonomics, the following features should be integrated:

| Feature | Where it helps | Effort |
| :--- | :--- | :--- |
| **bsuite-style diagnostic scenarios** | The core novel contribution for reviewers | Medium |
| **dm_env adapter** | Unlocks Acme / TF-Agents ecosystem out of the box | Tiny |
| **Multi-objective reward decomp** | Per-objective ablations (NetworkGym pattern) | Small |
| **Procedural difficulty curriculum** | Data-augmentation / generalization evaluation (ProcGen) | Small |
| **Structured per-episode logger** | Research ergonomics (W&B/TensorBoard ready, Acme pattern) | Small |
| **Spec objects (`dm_env.specs`)** | Typed observation/action contracts | Tiny |

---

## 3. Codebase Quality & Artifact Health

Reviewers demand executable, well-documented code with explicitly stated limitations and assumptions. The current codebase has strong foundations but several reviewer-facing artifact flaws.

### Strengths
*   **100% Green Test Suite:** 216/216 tests passing, proving parity between the legacy PyTorch backend and the vectorized JAX core.
*   **Architectural Maturity:** `main` (at `59c1e8e`) is in an incredibly advanced state, already integrating semantic LLM wrappers, Anthropic/OpenAI clients, and JAX CVE-gated exploits.
*   **PettingZoo API & Bridges:** CleanRL, Stable-Baselines3, and PettingZoo wrappers are present and tested.

### Weaknesses & Concrete Artifact Issues
> [!WARNING]
> **Hidden Heavy Dependencies:** The codebase utilizes `scapy`, `openai`, `anthropic`, `docker`, and `sentence-transformers`, but these are not accurately advertised in the `pyproject.toml` optional dependency groups (`dev`, `docs`, `jax`, `render`, `finetune`). "Executable and documented" is a hard requirement, and silent import failures will cause immediate rejection.

> [!CAUTION]
> **Broken Documentation Configurations:** The `mkdocs.yml` is still wired to old `xaiqo` URLs, and many linked `.md` pages in the config no longer exist in the `docs/` layout due to recent consolidations. This artifact-quality issue quietly undermines credibility.

> [!NOTE]
> **Linting Debt:** `ruff check .` exposes 44 active linting errors, almost entirely due to `pytest.importorskip('jax.numpy')` triggering `E402` (imports not at top of file) in the test suite.

---

## 4. The Risk Map (Where Reviewers Will Press)

Going into the submission, the following risk areas require mitigation:

1.  **Full Semantic Parity:** Does the vectorized JAX step mathematically guarantee parity with the legacy PyTorch step under all stochastic edge cases?
2.  **Benchmark Protocol Clarity:** The exact protocol for the Zero-Shot LLM leaderboard vs. the classical RL leaderboard must be standardized. Is the exact scenario, tick limit, and observation space clearly dictated?
3.  **Reproducibility Beyond One Golden Trajectory:** The `chore/audit-baseline` locks one golden fingerprint (`abd164a5...`), but reviewers will want to see generalization of reproducibility across varying procedural seeds and difficulty curriculums.
4.  **Roadmap Promises vs. Shipped Code:** The README and paper must strictly differentiate between what is currently executing in `main` and what is future work. Claiming LLM functionality that fails out-of-the-box due to missing prompt templates or undocumented API keys is a fatal risk.
