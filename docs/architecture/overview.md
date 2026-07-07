# Execution Architecture

## 1. Simulation Loop
The simulation loop is fully parallelized via the PettingZoo API (`parallel_env.step()`) and JAX vectorization (`jax.vmap`).

```mermaid
sequenceDiagram
    participant Agents
    participant Environment
    participant ActionRegistry
    participant ConflictResolutionEngine
    participant GlobalNetworkState

    Agents->>Environment: step({agent_id: [action_type, target_index]})
    Environment->>ActionRegistry: Instantiate actions
    Environment->>Environment: action.execute(GlobalNetworkState)
    Environment->>ConflictResolutionEngine: resolve(effects_dict)
    ConflictResolutionEngine->>GlobalNetworkState: Apply ActionEffect.state_deltas
    Environment-->>Agents: Return new observations, rewards, terminations
```

## 2. Telemetry Pipeline
Blue agent observations are generated exclusively through the simulated Security Information and Event Management (SIEM) pipeline. 

```mermaid
flowchart TD
    subgraph Action Execution
        R1[Red Actions]
    end

    subgraph Simulation Core
        GNS[(GlobalNetworkState)]
        CRE{Conflict Resolution Engine}
        SL[SIEMLogger]
    end

    subgraph Blue Observability
        LB[(siem_log_buffer)]
        NLP[NLP Log Encoder]
        BO[Blue Agent Observation Vector]
    end

    R1 -->|ActionEffect| CRE
    CRE -->|Valid Effects| GNS
    CRE -->|Action Metrics| SL
    
    SL -.->|Stochastic Noise Injection| SL
    SL -->|Generates Sysmon/Windows Logs| LB
    
    LB -->|N Most Recent Logs| NLP
    NLP -->|TF-IDF / Dense Embeddings| BO
```

## 3. Component Details

### `BaseAction` and `ActionEffect`
All agent capabilities inherit from `BaseAction`. `execute()` returns an `ActionEffect` containing `state_deltas` determining specific state tensor modifications.

### `ConflictResolutionEngine`
Resolves temporal collisions occurring in the same parallel tick. Handled deterministically within the `EnvState` interpreter.

### `SIEMLogger` and `LogEncoder`
`SIEMLogger` translates state deltas into standardized string logs matching Windows/Sysmon syntax, injecting stochastic benign noise. When `log_latency > 0` it holds each log for that many ticks before it becomes visible, modelling a lagging SOC feed. `LogEncoder` vectorizes the buffer into dense observations.
