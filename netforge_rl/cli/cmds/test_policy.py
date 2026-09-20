def cmd_test_policy(args) -> int:
    from netforge_rl.arena.agent_card import build_agent_card, format_agent_card
    from netforge_rl.baselines.registry import make_policy

    card = build_agent_card(
        lambda: make_policy(args.policy),
        name=args.policy,
        seeds=tuple(args.seeds),
        run_arena=not args.no_arena,
        out_dir=str(args.out) if args.out else None,
    )
    print(format_agent_card(card))
    return 0
