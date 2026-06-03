# Datasheet for NetForge-MARL

Following [Gebru et al., 2021](https://arxiv.org/abs/1803.09010) — required for the NeurIPS Datasets & Benchmarks track.

## Motivation

**For what purpose was the dataset created?** To provide a single common substrate for multi-agent reinforcement learning research on cybersecurity that (a) preserves the domain fidelity practitioners need (SIEM telemetry, MITRE ATT&CK, ZTNA, OT/ICS kinetic impact) and (b) delivers the throughput modern algorithms require. NetForge-MARL is also designed to be the first MARL benchmark in which language and vision foundation models are first-class agents.

**Who created the dataset and on behalf of which entity?** The ReforceMind team. Independent research.

**Who funded the dataset's creation?** Self-funded.

## Composition

**What do the instances that comprise the dataset represent?** Procedurally generated three-tier enterprise networks (DMZ / Corporate / Secure ± OT) of exactly 100 host slots (active topology spans 15–30 nodes; the remainder are pinned link-local padding for tensor-shape stability). Each scenario is a Python class with reward and termination logic.

**How many instances are there in total?** Infinite — the topology generator produces a new instance per `(scenario, seed)`. The repo ships **2 reference scenarios** (`ransomware`, `apt_espionage`) at v4.0; Phase 5 will expand to 5 standardized scenarios.

**What data does each instance consist of?** A `GlobalNetworkState` (or its frozen `EnvState` mirror) containing host attributes (status, privilege, OS, services, vulnerabilities, ZTNA tokens, decoy flags, honeytokens), subnet topology, firewall rules, agent inventories / fog-of-war, and per-tick action history.

**Is there a label or target?** Reward signals per agent are defined by the scenario; canonical metrics emitted by every episode: Mean Time To Containment (MTTC), SLA Uptime, Total Exfiltrated Data, hosts compromised / isolated.

**Are there recommended data splits?** Train: any seeds. Eval: a held-out seed range published with each NeurIPS submission (TBD); zero-shot foundation-model leaderboard uses seeds 0..49 of each scenario.

**Are there any errors, sources of noise, or redundancies in the dataset?** Synthetic — by construction. The `pcap_synthesizer` produces structurally realistic packets but is not derived from real network traces.

## Collection process

**How was the data associated with each instance acquired?** Procedural generation from `netforge_rl/topologies/network_generator.py` (seeded `random.Random`).

**What mechanisms or procedures were used to collect the data?** N/A — fully synthetic.

**Does the dataset relate to people?** No.

## Preprocessing / cleaning / labeling

**Was any preprocessing of the data done?** Padding hosts (`169.254.0.0/16`) are appended to every instance to keep the host tensor at exactly 100 rows. The frozen `EnvState` representation encodes categorical attributes (`status`, `privilege`, `decoy`) against pinned codebooks documented in `netforge_rl/core/functional.py`.

## Uses

**Has the dataset been used for any tasks already?** RL training of attack / defense policies; reference baselines in `netforge_rl/baselines/`; zero-shot foundation-model evaluation via `netforge_rl/semantic/runner.py`; LoRA + PPO fine-tuning of open-weights LLMs (notebook `07_finetune_llama3_lora.ipynb`).

**Is there a repository that links to any or all papers or systems that use the dataset?** Yes — the leaderboard at `leaderboard/results.json` (under construction).

**What (other) tasks could the dataset be used for?** Sim-to-real transfer studies (the Docker-backed `Sim2RealBridge` evaluates trained policies against live Vulhub containers); causal-reasoning probes for LLM agents in incident response; offline RL on logged trajectories.

**Is there anything about the composition of the dataset that might impact future uses?** Action taxonomies are aligned to MITRE ATT&CK as of February 2026; later additions to ATT&CK will need to be ported.

**Are there tasks for which the dataset should not be used?** Generating exploits for live use against unauthorized systems. The Vulhub `Sim2RealBridge` is intended for evaluating defensive policies against intentionally vulnerable lab images only.

## Distribution

**Will the dataset be distributed to third parties outside of the entity?** Yes — open source under the MIT License.

**How will the dataset be distributed?** Source via GitHub (`reforcemind/NetForge_RL`); a frozen v4.0.0 release archive will be deposited to Zenodo with a DOI for the NeurIPS submission.

## Maintenance

**Who is supporting / hosting / maintaining the dataset?** ReforceMind via the GitHub issue tracker.

**Will the dataset be updated?** Yes — Phase 5 expands the scenario suite; Phase 8 milestones M2 / M3 will add leaderboard results.

**If others want to extend / augment / build on / contribute to the dataset, is there a mechanism for them to do so?** Pull requests against `main`, gated on (1) full test suite passing, (2) golden trajectory hash preserved or re-locked with a changelog entry, (3) SPS regression ≤ 5 % vs the previous tagged release.
