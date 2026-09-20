from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='netforge',
        description='Cybersecurity gym: run, evaluate, diagnose.',
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_run = sub.add_parser('run', help='Run a scenario YAML or named pack')
    p_run.add_argument('scenario', help='Pack name or path to .yaml')
    p_run.add_argument('--policy', default='heuristic-blue')
    p_run.add_argument('--red', default='killchain-red')
    p_run.add_argument('--seed', type=int, default=0)
    p_run.add_argument('--max-ticks', type=int, default=80)
    p_run.add_argument('--replay', type=Path, default=None)

    p_test = sub.add_parser('test-policy', help='Capability tests + agent card')
    p_test.add_argument('--policy', default='heuristic-blue')
    p_test.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    p_test.add_argument('--out', type=Path, default=None)
    p_test.add_argument('--no-arena', action='store_true')

    p_bench = sub.add_parser('benchmark', help='Arena evaluation / baseline table')
    p_bench.add_argument('policy', nargs='?', default='heuristic-blue')
    p_bench.add_argument('--split', default='dev', choices=['train', 'dev', 'hidden'])
    p_bench.add_argument('--red', default='killchain-red')
    p_bench.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    p_bench.add_argument('--max-ticks', type=int, default=40)
    p_bench.add_argument('--scenarios', nargs='+', default=['ransomware'])
    p_bench.add_argument('--table', action='store_true', help='Run the reference table')
    p_bench.add_argument('--out', type=Path, default=None)

    p_arena = sub.add_parser('arena', help='Print Arena spec or run a split')
    p_arena.add_argument('--split', default=None, choices=['train', 'dev', 'hidden'])
    p_arena.add_argument('--policy', default='heuristic-blue')
    p_arena.add_argument('--red', default='killchain-red')
    p_arena.add_argument('--json', action='store_true')

    p_rep = sub.add_parser('replay', help='Write an interactive HTML replay')
    p_rep.add_argument('trajectory', type=Path)
    p_rep.add_argument('--out', type=Path, default=Path('replay.html'))

    p_ds = sub.add_parser('collect', help='Collect an offline-RL dataset')
    p_ds.add_argument('--policy', default='heuristic-blue')
    p_ds.add_argument('--red', default='random')
    p_ds.add_argument('--episodes', type=int, default=4)
    p_ds.add_argument('--max-ticks', type=int, default=30)
    p_ds.add_argument('--quality', default='mixed')
    p_ds.add_argument('--out', type=Path, default=Path('netforge_offline.npz'))
    p_ds.add_argument(
        '--hdf5',
        action='store_true',
        help='Write Minari-like HDF5 (needs h5py) instead of npz',
    )

    p_q = sub.add_parser('questions', help='Catalog / run named RL probes')
    p_q.add_argument(
        'question', nargs='?', default=None, help='Question id, or omit to list'
    )
    p_q.add_argument('--all', action='store_true', help='Run every runnable question')
    p_q.add_argument('--json', action='store_true')
    p_q.add_argument('--seeds', type=int, nargs='+', default=[0])
    p_q.add_argument('--max-ticks', type=int, default=16)

    p_ev = sub.add_parser('evaluate', help='CAGE-style submission eval (multi-metric)')
    p_ev.add_argument('submission', type=Path, help='Python module with make_blue()')
    p_ev.add_argument('--split', default='dev', choices=['train', 'dev', 'hidden'])
    p_ev.add_argument('--seeds', type=int, nargs='+', default=None)
    p_ev.add_argument('--max-ticks', type=int, default=None)
    p_ev.add_argument('--scenarios', nargs='+', default=None)
    p_ev.add_argument('--official', action='store_true', help='Arena 2026 freeze')
    p_ev.add_argument('--card', action='store_true', help='Also run capability probes')
    p_ev.add_argument('--out', type=Path, default=None)

    sub.add_parser('version', help='Print package / Arena version')
    return parser
