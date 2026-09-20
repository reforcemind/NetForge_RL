from __future__ import annotations

import json

from netforge_rl.arena.spec import ARENA_NAME, current_arena


def cmd_arena(args) -> int:
    spec = current_arena()
    if args.split:
        from netforge_rl.arena.evaluate import evaluate_split
        from netforge_rl.baselines.registry import make_policy

        result = evaluate_split(
            lambda: make_policy(args.red),
            lambda: make_policy(args.policy),
            split=args.split,
            scenarios=('ransomware',),
            seeds=(0, 1),
            max_ticks=40,
        )
        print(json.dumps(result, indent=2))
        return 0
    payload = spec.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(ARENA_NAME)
        print(json.dumps(payload, indent=2))
    return 0
