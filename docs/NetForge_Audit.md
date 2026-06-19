# NetForge RL Code Review & Audit

This audit evaluates the codebase of the `NetForge_RL` multi-agent simulator, focusing on logic flaws, dead code, reward exploits, and its potential as a benchmark submission for NeurIPS.

## 1. Critical Logic Bugs & Reward Exploits

### 1.1 Unbounded Lateral Movement (JAX Backend)
In `netforge_rl/backends/jax/vector_env.py`, the `PassTheHash` (PTH) and `PassTheTicket` (PTT) actions are universally overpowered. 
The gating logic checks if the agent holds *any* token (`agent_has_any_token`), rather than checking if the token is valid for the target host:
```python
# vector_env.py:173
red_writes_user_pth = jnp.any(
    red_target_mask & (red_is_pth & red_token)[:, None], axis=0
)
```
**Impact:** If a Red agent dumps LSASS and loots a token from *one* machine, they can instantly bypass authentication on *any* other host in the network. Token locality and validity matching is entirely missing.

### 1.2 Reward Farming Vulnerabilities (JAX Backend)
Several reward calculations are not gated on the target's current state, allowing agents to farm points infinitely without altering the environment:
- **Blue Isolate:** Blue can repeatedly queue `BLUE_ISOLATE` on an *already isolated* host and receive a `+1.0` team reward every tick.
- **Red Impact:** Red can repeatedly spam `RED_IMPACT` on an *already impacted* (Rooted) host for a constant `+10.0` reward per tick.
**Fix:** Mask the reward assignments based on whether the action caused a state transition (e.g., `(new_status != old_status) & blue_writes_isolate`).

### 1.3 Premature Action Cancellation (Legacy Backend)
In `netforge_rl/environment/parallel_env.py`, when a Blue agent issues an `IsolateHost` action, the environment immediately cancels all in-flight Red actions against that target:
```python
# parallel_env.py:211
if type(event['action']).__name__ == 'IsolateHost' and event['completion_tick'] > self.current_tick:
    # Immediately removes Red events and unlocks the Red agent
```
**Impact:** Blue actions instantly "stun" Red agents the moment they are queued, rather than when the isolation *completes*. This effectively gives Blue a zero-latency counter-attack, breaking the temporal simulation.

## 2. Wasted Code & Broken Metrics

### 2.1 Unpopulated Telemetry Metrics
In `parallel_env.py`, you initialize `episode_metrics` for `infection_times`, `isolation_times`, and `sla_uptime_sum`. However, **these dictionaries are never updated during `step()`**.
```python
# parallel_env.py:462
sla_final = self.episode_metrics['sla_uptime_sum'] / self.episode_metrics['steps_count']
```
**Impact:** `SLA_Uptime_Percentage` and `MTTC` (Mean Time to Containment) will always output `0.0` or `1.0`. The metrics tracking logic needs to be wired into `_extract_agent_infos()` or `_apply_state_deltas()`.

### 2.2 Global Blue Agent Concurrency Bottleneck
`blue_active_actions_count` caps the number of concurrent Blue actions to exactly 2:
```python
# parallel_env.py:183
if 'blue' in agent.lower():
    if blue_active_actions_count >= 2:
        continue
```
**Impact:** Because this is evaluated across the entire event queue, it restricts the *whole* Blue team (e.g., `blue_dmz`, `blue_internal`, `blue_restricted`) to 2 actions globally, negating the advantage of decentralized MARL operators. It should be bounded per-agent or per-subnet.

### 2.3 `ruff` Linting Errors
Running `ruff check .` surfaces 42 errors, primarily consisting of unused imports (e.g., `import jax.numpy as jnp` in `test_state_and_kernels.py`) and module-level imports located below `pytest.importorskip` statements.

## 3. NeurIPS Benchmark Potential

**Verdict: Extremely High Potential**
NetForge possesses several compelling attributes for a NeurIPS Datasets and Benchmarks track submission:
1. **Unprecedented Throughput:** Operating at 1M+ SPS on a CPU via `jax.vmap` while maintaining a frozen PyTree state is state-of-the-art for MARL cybersecurity simulators (significantly outperforming CybORG).
2. **"Semantic Bridge" Modality:** Supporting Foundation Models (via `trl` LoRA fine-tuning) as first-class agents interacting directly with `JaxMARL` policies is a massive differentiator.
3. **Dual-Backend Parity:** Offering both a PettingZoo/Sim2Real functional core and a vector-accelerated JAX core provides both research scale and applied deployment realism.

### Recommended Improvements Prior to Submission
- **Fix the JAX State Gating:** Address the reward farming and unbounded `PassTheHash` logic. Reviewers will write custom policies, and MAPPO/PPO will instantly find and exploit these local optima.
- **Differentiate Kerberos Rotation:** `BLUE_ROTATE_KERBEROS` instantly wipes all credentials (`new_red_creds = jnp.zeros_like...`) globally. Introduce domain vs. local token distinctions to prevent Blue from solving the game with a single global button.
- **Metrics Fix:** Hook up the MTTC and SLA metric pipelines. Reviewers heavily value built-in evaluation heuristics beyond simple scalar rewards.
