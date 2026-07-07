# NetForge RL — Environment Specification

Agents (4): red_operator, blue_dmz, blue_internal, blue_restricted

Action types: 20 red, 14 blue.

## Action space

| Agent | Type | nvec | Meaning |
|---|---|---|---|
| red_operator | MultiDiscrete | [32, 100] | [action_type_id, target_host_index] |
| blue_dmz | MultiDiscrete | [32, 100] | [action_type_id, target_host_index] |
| blue_internal | MultiDiscrete | [32, 100] | [action_type_id, target_host_index] |
| blue_restricted | MultiDiscrete | [32, 100] | [action_type_id, target_host_index] |

## Observation space (per agent, Dict)

| Key | Shape | dtype | range |
|---|---|---|---|
| action_mask | [132] | int8 | [0.0, 1.0] |
| adj_matrix | [10000] | float32 | [0.0, 1.0] |
| delta_t | [1] | float32 | [0.0, 1.0] |
| obs | [256] | float32 | [-1.0, 1.0] |
| siem_embedding | [128] | float32 | [-1.0, 1.0] |

## Observability model

| Agent | View |
|---|---|
| red_operator | Fog of war — only hosts discovered via recon are visible; host compromise state for owned hosts. |
| blue_dmz | SIEM-filtered view of the DMZ subnet plus shared blue channel. |
| blue_internal | SIEM-filtered view of internal subnets plus shared blue channel. |
| blue_restricted | SIEM-filtered view of restricted/Secure subnets plus shared blue channel. |

## Termination

| Scenario | Condition |
|---|---|
| ransomware | every Corporate/Secure host has system_integrity=compromised, or any PLC reaches kinetic_destruction. |
| apt_espionage | every infected host has been isolated (blue contains the breach). |
| cloud_hybrid | every Secure-subnet host is compromised. |
| iot_grid | all controllers (domain controllers) compromised. |
| ot_stuxnet | any PLC reaches kinetic_destruction. |
| * | truncation at max_ticks (default 200) for every scenario. |

## Reward decomposition (bounded, tanh-squashed)

### ransomware (MAX_STEP_REWARD=5.0)
- red: privilege_user=3.0, privilege_root=5.0, system_compromised=10.0, kinetic_destruction=10000.0, host_owned=2.0, host_discovered=0.5, scan_result=0.3, intel_shared=1.0, hit_decoy=-3.0, kernel_panic=-5.0, action_cost=-0.05, failed_action=-0.1
- blue: correct_isolation=5.0, false_positive_isolation=-2.0, host_cleaned=3.0, service_restored=1.0, decoy_deployed=1.0, kinetic_failure=-10000.0, health_ratio=1.0, compromised_ratio=-2.0, business_downtime=-5.0, action_cost=-0.05, failed_action=-0.1

### apt_espionage (MAX_STEP_REWARD=20.0)
- red: recon=2.0, breach=10.0, exfiltration=20.0, intel_shared=5.0, hit_decoy=-15.0, persistence_online=5.0, persistence_isolated=-20.0, action_cost=-0.1
- blue: host_clean=1.0, host_infected=-5.0, infected_isolated=20.0, action_cost=-0.1

### iot_grid (MAX_STEP_REWARD=250.0)
- red: privilege=2.0, controller_breach=40.0, kernel_panic=5.0, failed_action=-0.1, action_cost=-0.05
- blue: health_ratio_scale=0.5, controller_loss=-250.0, correct_isolation=3.0, false_positive_isolation=-1.0, action_cost=-0.05

### ot_stuxnet (MAX_STEP_REWARD=10000.0)
- red: ot_privilege=12.0, it_privilege=4.0, kinetic_destruction=10000.0, failed_action=-0.1, action_cost=-0.05
- blue: ot_hold_bonus=0.5, kinetic_penalty=-10000.0, ot_isolation=6.0, action_cost=-0.05

### cloud_hybrid (MAX_STEP_REWARD=30.0)
- red: secure_breach=25.0, dmz_breach=1.0, internal_breach=2.0, failed_action=-0.1, action_cost=-0.05
- blue: secure_sla_scale=1.0, secure_loss=-30.0, secure_isolation=5.0, compromised_isolation=1.5, false_positive_isolation=-1.0, action_cost=-0.05
