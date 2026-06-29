from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from netforge_rl.backends.jax.vector_env import (
    N_BLUE_ACTIONS,
    N_RED_ACTIONS,
    SCENARIO_IDS,
    get_reward_weights,
)
from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.scenarios import _SCENARIOS

RESULTS_DIR = Path(__file__).parent / 'results'

OBSERVABILITY = {
    'red_operator': 'Fog of war — only hosts discovered via recon are visible; '
    'host compromise state for owned hosts.',
    'blue_dmz': 'SIEM-filtered view of the DMZ subnet plus shared blue channel.',
    'blue_internal': 'SIEM-filtered view of internal subnets plus shared blue channel.',
    'blue_restricted': 'SIEM-filtered view of restricted/Secure subnets plus shared '
    'blue channel.',
}

TERMINATION = {
    'ransomware': '>=90% of hosts compromised, or any PLC kinetic destruction.',
    'apt_espionage': 'every infected host has been isolated (blue contains the breach).',
    'cloud_hybrid': 'all critical (domain-controller) hosts compromised.',
    'iot_grid': 'all controllers (domain controllers) compromised.',
    'ot_stuxnet': 'any PLC reaches kinetic_destruction.',
    '*': 'truncation at max_ticks (default 200) for every scenario.',
}


def _space_repr(space) -> dict:
    out = {}
    for key, sub in space.spaces.items():
        out[key] = {
            'shape': list(sub.shape),
            'dtype': str(sub.dtype),
            'low': float(np.min(sub.low)),
            'high': float(np.max(sub.high)),
        }
    return out


def build_spec() -> dict:
    env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 200})
    env.reset(seed=0)
    agents = env.possible_agents

    obs = {a: _space_repr(env.observation_space(a)) for a in agents}
    act = {
        a: {
            'type': 'MultiDiscrete',
            'nvec': list(map(int, env.action_space(a).nvec)),
            'meaning': '[action_type_id, target_host_index]',
        }
        for a in agents
    }
    reward = {
        name: {
            'max_step_reward': cls.MAX_STEP_REWARD,
            **get_reward_weights(name),
        }
        for name, cls in _SCENARIOS.items()
        if name in SCENARIO_IDS
    }
    return {
        'agents': list(agents),
        'n_red_action_types': N_RED_ACTIONS,
        'n_blue_action_types': N_BLUE_ACTIONS,
        'observation_spaces': obs,
        'action_spaces': act,
        'observability': OBSERVABILITY,
        'reward_decomposition': reward,
        'termination': TERMINATION,
    }


def to_markdown(spec: dict) -> str:
    L = [
        '# NetForge RL — Environment Specification',
        '',
        f'Agents ({len(spec["agents"])}): ' + ', '.join(spec['agents']),
        '',
        f'Action types: {spec["n_red_action_types"]} red, '
        f'{spec["n_blue_action_types"]} blue.',
        '',
        '## Action space',
        '',
        '| Agent | Type | nvec | Meaning |',
        '|---|---|---|---|',
    ]
    for a, v in spec['action_spaces'].items():
        L.append(f'| {a} | {v["type"]} | {v["nvec"]} | {v["meaning"]} |')

    L += [
        '',
        '## Observation space (per agent, Dict)',
        '',
        '| Key | Shape | dtype | range |',
        '|---|---|---|---|',
    ]
    for key, v in next(iter(spec['observation_spaces'].values())).items():
        L.append(f'| {key} | {v["shape"]} | {v["dtype"]} | [{v["low"]}, {v["high"]}] |')

    L += ['', '## Observability model', '', '| Agent | View |', '|---|---|']
    for a, desc in spec['observability'].items():
        L.append(f'| {a} | {desc} |')

    L += ['', '## Termination', '', '| Scenario | Condition |', '|---|---|']
    for s, c in spec['termination'].items():
        L.append(f'| {s} | {c} |')

    L += ['', '## Reward decomposition (bounded, tanh-squashed)', '']
    for name, r in spec['reward_decomposition'].items():
        L.append(f'### {name} (MAX_STEP_REWARD={r["max_step_reward"]})')
        L.append('- red: ' + ', '.join(f'{k}={v}' for k, v in r['red_weights'].items()))
        L.append(
            '- blue: ' + ', '.join(f'{k}={v}' for k, v in r['blue_weights'].items())
        )
        L.append('')
    return '\n'.join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--json', action='store_true', help='also write results/env_spec.json'
    )
    args = p.parse_args()
    spec = build_spec()
    md = to_markdown(spec)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print(md)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'env_spec.md').write_text(md, encoding='utf-8')
    if args.json:
        (RESULTS_DIR / 'env_spec.json').write_text(json.dumps(spec, indent=2))


if __name__ == '__main__':
    main()
