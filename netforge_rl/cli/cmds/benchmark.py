from __future__ import annotations

import json


def cmd_benchmark(args) -> int:
    from netforge_rl.arena.evaluate import evaluate_split
    from netforge_rl.arena.leaderboard import format_leaderboard, save_leaderboard
    from netforge_rl.baselines.protocol import format_baseline_table, run_baseline_table
    from netforge_rl.baselines.registry import make_policy

    if args.table:
        table = run_baseline_table(
            seeds=tuple(args.seeds),
            scenarios=tuple(args.scenarios),
            max_ticks=args.max_ticks,
            split=args.split,
        )
        print(format_baseline_table(table))
        if args.out:
            args.out.write_text(json.dumps(table, indent=2))
        return 0
    result = evaluate_split(
        lambda: make_policy(args.red),
        lambda: make_policy(args.policy),
        split=args.split,
        scenarios=tuple(args.scenarios),
        seeds=tuple(args.seeds),
        max_ticks=args.max_ticks,
    )
    result['name'] = args.policy
    print(json.dumps(result['overall'], indent=2))
    if args.out:
        save_leaderboard([result], args.out)
        print(format_leaderboard([result]))
    return 0
