from __future__ import annotations

import json


def cmd_evaluate(args) -> int:
    from netforge_rl.arena.submissions import evaluate_submission, load_submission

    sub = load_submission(args.submission)
    result = evaluate_submission(
        sub,
        split=args.split,
        scenarios=args.scenarios,
        seeds=args.seeds,
        max_ticks=args.max_ticks,
        official=args.official,
        agent_card=args.card,
    )
    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0
