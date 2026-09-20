from __future__ import annotations

import json


def cmd_questions(args) -> int:
    from netforge_rl.arena.questions import (
        format_catalog,
        list_questions,
        run_all_runnable,
        run_question,
    )

    if args.all:
        payload = run_all_runnable(seeds=tuple(args.seeds), max_ticks=args.max_ticks)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.question:
        payload = run_question(
            args.question, seeds=tuple(args.seeds), max_ticks=args.max_ticks
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.json:
        print(json.dumps(list_questions(), indent=2))
    else:
        print(format_catalog())
    return 0
