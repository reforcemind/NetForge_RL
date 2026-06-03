# NetForge-MARL — Roadmap (NeurIPS Datasets & Benchmarks 2026)

This is the canonical planning document. Phase 0 audit findings, measured
SPS deltas, and regression locks live in [AUDIT.md](AUDIT.md); per-phase
implementation lives on the named feature branches.

---

## Phase 0 — Audit & baseline lock-in ✅
**Branch:** `chore/audit-baseline`

Pinned the legacy backend's SPS baseline (~10 SPS, single env) and locked
the golden trajectory fingerprint behind a regression test. Every later
phase asserts the legacy hash stays stable or documents a deliberate
re-lock.

## Phase 1 — Functional core ✅
**Branch:** `refactor/functional-core`

Frozen `EnvState` PyTree, pure `apply_state_delta` / `resolve_conflicts`,
episode-long parity proof against the legacy step. Legacy backend
untouched; new functional primitives are JAX-ready by construction.

## Phase 2 — JAX backend (slices 1–2 ✅, 3+ pending)
**Branch:** `feat/jax-core`

Registered PyTree state, vectorizable kernels, `jax.vmap` batched step,
1,082,255 aggregate SPS at 4096 envs on CPU (103,500× speedup over the
single-env baseline). Slice 3 will port the remaining 32 actions; slice 4
will close numerical parity with the legacy step under a configurable
tolerance for stochastic effects.

## Phase 3 — API standardization
**Branches:** `feat/api-jaxmarl`, `feat/api-pettingzoo`, `feat/bridges-dlpack`

JaxMARL-shape adapter (`reset(key) -> (obs, state)`, `step(key, state, actions)`)
on the JAX backend; PettingZoo `ParallelEnv` on the PyTorch backend; zero-copy
DLPack converters so CleanRL / Stable-Baselines3 / RLlib can consume the
vmap'd rollouts without leaving GPU.

## Phase 4 — Decoupled rendering
**Branch:** `feat/render-pipeline`

NetworkX layout + Matplotlib/Pygame renderers driven off frozen `EnvState`
snapshots — never touches the hot path. `RecorderWrapper` emits `.mp4` /
`.gif` evaluation reels via moviepy; CI auto-attaches a 30s gif to every PR.

## Phase 5 — Baselines, scenarios, leaderboard
**Branches:** `feat/scenarios-suite`, `feat/baseline-{mappo,ippo,qmix,ppo}`

Five standardized scenarios (`enterprise-it-{small,large}`, `iot-grid`,
`ot-stuxnet`, `cloud-hybrid`); reference baselines in JAX (MAPPO/IPPO/QMIX)
and PyTorch (PPO/MAPPO); a public leaderboard auto-deployed to GitHub Pages.

## Phase 6 — Research readiness ("WOW" docs)
**Branch:** `docs/neurips-ready`

Colab tutorial suite (notebooks 01–06: quickstart, attack viz, CleanRL,
MAPPO@4k envs, Sim2Real, custom scenarios); MkDocs site overhaul with
citations + MITRE coverage matrix; Datasheet for Datasets + NeurIPS D&B
checklist; hero animation in README.

## Phase 7 — Submission hardening
**Branch:** `release/v4.0-neurips`

Frozen `v4.0.0` tag, Zenodo DOI, camera-ready paper, public leaderboard
live with ≥3 submitted methods.

---

## Phase 8 — Semantic Bridge: Foundation-Model-Native MARL 🆕

This pillar reframes NetForge from "a MARL benchmark with optional LLM
hooks" to **the first MARL benchmark in which language and vision models
are first-class agents**, alongside PPO/MAPPO/QMIX. It lands after the
JAX backend (Phase 2) and the decoupled renderer (Phase 4), both of
which are prerequisites. Phase 8 has its own dual-backend logic: the
text wrappers ride on either backend; the fine-tuning loop is
PyTorch-only because it depends on the Hugging Face `trl` stack.

### 8.1 The Language-Action Wrapper (LA)

A bidirectional, real-time translator between numerical environment state
and natural-language tokens.

**Forward path (env → LLM):** intercepts the dictionary-of-arrays
observation that the env emits and produces a concise, structured
"SIEM telemetry report" — readable strings such as *"Critical Alert:
SSH brute-force originating on `node_42` against `node_07`; privilege
elevated to User on `node_15` (last 3 ticks); 2 hosts isolated in DMZ."*
Reports are templated per agent role (Blue SOC operator vs Red operator)
and trimmed to fit a configurable token budget so context-window-limited
models (8k → 200k) all work out of the box.

**Reverse path (LLM → env):** parses the model's text output into one of
the env's discrete `(action_type, target)` action IDs. Two parsers ship
in v1: a strict regex parser (`ISOLATE host_42`) and a logit-bias parser
that constrains generation to the legal action grammar via the OpenAI
`tool_choice` / Anthropic `tool_use` / vLLM grammar-constrained-decoding
hooks — eliminating the "LLM hallucinates an invalid action" failure mode
that dominates existing LLM-RL benchmarks.

### 8.2 The Vision-Language-Action Wrapper (VLA)

Sits on top of the Phase 4 renderer. Each tick, the wrapper grabs the
rendered RGB frame (network topology where node colors encode state:
green = secure, red = compromised, yellow = honeytoken, blue = defended,
grey = isolated) and pairs it with the LA-generated SIEM report. The
combined `(image, text)` prompt is then dispatched to multimodal
models — LLaVA, GPT-4o, Claude Sonnet vision, Gemini Pro Vision — which
can literally *see* the attack surface evolving alongside reading its
log telemetry. This is the first MARL benchmark we are aware of where
vision is a first-class modality of the policy, not just a debugging
artifact.

### 8.3 Zero-Shot Foundation-Model Benchmarking

A reference client harness for closed-source APIs (Anthropic, OpenAI,
Google) that runs an entire episode without any training — pure
in-context reasoning. Output: a standardized scoreboard reporting Mean
Time To Containment (MTTC), False-Positive rate, SLA Uptime, and Total
Exfiltrated Data for each (model, scenario) pair. Three scoreboard
classes ship:

* **Zero-Shot Defender** (model controls Blue against the scripted B-line Red).
* **Zero-Shot Attacker** (model controls Red against a heuristic SOC).
* **Foundation-Model vs Foundation-Model** (e.g. GPT-4o Blue vs Claude Sonnet Red).

This becomes "the first standardized leaderboard for LLM-driven Incident
Response" — a result that is itself publishable and a strong magnet for
external submissions.

### 8.4 Native LLM Fine-Tuning (LoRA + PPO)

Integration with `trl` so open-weights models (Llama-3-8B, Qwen2-7B,
Mistral-7B-Instruct) can be fine-tuned by **playing the environment**.
Specifically:

* Reference recipe for `PPOTrainer` with a 4-bit quantized policy +
  LoRA adapters — fits on a single 24 GB consumer GPU.
* `LMPolicyAdapter` translates between the env's PettingZoo step
  interface and `trl`'s `(query, response, reward)` rollout protocol.
* Rewards forwarded directly from the env's existing scenario logic
  (MTTC, exfil penalty, SLA bonus, kinetic-impact super-reward) — no
  separate reward model needed, which is the usual RLHF bottleneck.

### 8.5 Milestones

| # | Title | Deliverable | Dependencies | Exit criterion |
|---|---|---|---|---|
| 8-M1 | **Semantic Wrappers** | `netforge/semantic/{la_wrapper,vla_wrapper}.py`; templates per agent role; configurable token-budget trimmer; legal-action grammar parser. | Phase 2 (state extraction), Phase 4 (RGB frames). | Round-trip test: env → text → LLM-format → parsed action ID; vision wrapper emits a valid `(image, text)` pair every tick. |
| 8-M2 | **Zero-Shot Incident-Response Leaderboard** | API client harness; results JSON + GitHub-Pages scoreboard; ≥4 closed-source models scored on ≥3 scenarios. | 8-M1 + Phase 5 scenarios. | Public leaderboard live; results reproducible from a single `make leaderboard-zs` command + `.env` credentials. |
| 8-M3 | **Fine-Tuning Demonstrator** | `notebooks/07_finetune_llama3_lora.ipynb` — flagship Colab that fine-tunes Llama-3-8B-Instruct on `enterprise-it-small` with LoRA + PPO in < 4 h on a single 24 GB GPU; produces a checkpoint that measurably outperforms its base model on held-out scenarios. | 8-M1; Phase 5 baselines for comparison. | Notebook runs end-to-end on Colab Pro; LoRA-tuned policy beats base model on MTTC by a documented margin; checkpoint published to Hugging Face Hub under `reforcemind/netforge-llama3-blue-v1`. |

### 8.6 Where it lives in the architecture

```
netforge/
├── backends/                            # unchanged
│   ├── jax/
│   └── torch/
├── render/                              # Phase 4 — feeds VLA
│   └── ...
└── semantic/                            # 🆕 Phase 8
    ├── la_wrapper.py                    # env <-> text translator
    ├── vla_wrapper.py                   # env <-> (image, text) translator
    ├── templates/
    │   ├── soc_blue.j2                  # Blue SOC operator persona
    │   ├── red_operator.j2              # Red persona
    │   └── ot_engineer.j2               # OT/ICS persona (kinetic scenarios)
    ├── grammars/
    │   ├── anthropic_tools.json         # tool-use schemas
    │   ├── openai_tools.json
    │   └── vllm_grammar.lark            # constrained-decoding grammar
    ├── clients/                         # zero-shot harnesses
    │   ├── anthropic_client.py
    │   ├── openai_client.py
    │   ├── google_client.py
    │   └── vllm_client.py               # local OSS models
    └── finetune/                        # LoRA + PPO
        ├── lm_policy_adapter.py         # trl <-> PettingZoo bridge
        ├── ppo_recipe.py                # reference training script
        └── configs/
            └── llama3_8b_lora.yaml

notebooks/
└── 07_finetune_llama3_lora.ipynb        # 8-M3 flagship

baselines/
└── llm/                                 # zero-shot + fine-tuned leaderboard entries
```

### 8.7 Why this strengthens the NeurIPS submission

* **Distinct narrative arc.** Existing MARL benchmarks (JaxMARL, SMAX,
  Melting Pot) target classical algorithms only; existing LLM agent
  benchmarks (AgentBench, OSWorld, SWE-bench) are single-agent and not
  MARL-shaped. NetForge's dual axis (multi-agent × foundation-model)
  is uncontested.
* **Reviewer-bait artifact.** A Colab that fine-tunes Llama-3-8B to
  defend a real-ish network in < 4 h is a screenshot in the abstract.
* **Reproducibility flywheel.** The zero-shot leaderboard creates an
  ongoing external contribution path that survives past the camera-
  ready deadline — researchers don't need to train anything to
  participate.
* **Sim2Real bridge applies here too.** The Phase 0 inventory already
  includes a Docker-backed Sim2Real bridge; the fine-tuned LLM policies
  can be evaluated against live Vulhub containers as a transfer-
  generalization test, which is itself a publishable artifact.

### 8.8 Updated NeurIPS abstract — pitch v2

*Throttled progress on multi-agent reinforcement learning for cybersecurity
stems from a structural trade-off: physically faithful simulators run at
hundreds of steps per second and bar large-scale experimentation, while
throughput-optimized environments lack the partial observability, SIEM
telemetry, MITRE ATT&CK alignment, zero-trust enforcement, and OT/ICS
kinetic impact the security research community actually needs.
**NetForge-MARL** resolves the trade-off through a **dual-backend
architecture** — a JAX backend exploiting `jax.vmap` and XLA fusion for
> 10⁶ aggregate environment steps per second across 4,096 parallel
instances on a single accelerator, alongside a PyTorch / PettingZoo
backend with zero-copy DLPack interoperability for the broader RL
ecosystem. **More importantly, NetForge-MARL is the first cybersecurity
MARL benchmark in which foundation models are first-class agents.** A
Semantic Bridge layer translates between numerical state and structured
natural-language SIEM reports; a Vision-Language-Action wrapper pairs
those reports with rendered topology frames so multimodal models can
literally see the battlefield evolving; a reference client harness
produces the first standardized zero-shot leaderboard for LLM-driven
Incident Response across the leading closed-source APIs; and a `trl`-
based fine-tuning recipe lets researchers train an 8B parameter open-
weights model to defend an enterprise network in under four hours on a
single consumer GPU. We argue this dual axis — multi-agent × foundation
model — defines the missing common substrate for the next generation
of MARL cybersecurity research.*

---

## Branching strategy (extended)

```
main                                   protected, signed tags only
├── release/v4.0-neurips               release candidates
├── refactor/*                         internal restructure
├── feat/jax-*                         JAX backend
├── feat/torch-*                       PyTorch backend
├── feat/api-*                         standardization adapters
├── feat/render-*                      visualization
├── feat/scenarios-*                   new scenarios
├── feat/semantic-{la,vla,clients}     🆕 Phase 8 wrappers
├── feat/finetune-llm                  🆕 Phase 8 LoRA+PPO recipe
├── docs/*                             notebooks, mkdocs, roadmap
└── chore/*                            CI, lint, deps
```

Merge gates (cumulative): (1) full test suite green, (2) golden
trajectory hash matches or is re-locked with `CHANGELOG.md` entry, (3)
SPS regression ≤ 5 % relative to previous tagged release, (4) for Phase
8 PRs: a recorded `(image, text, action)` triplet demonstrating the
wrapper end-to-end on a real episode.
