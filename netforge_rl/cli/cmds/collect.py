def cmd_collect(args) -> int:
    from netforge_rl.baselines.registry import make_policy
    from netforge_rl.datasets.collector import collect_dataset
    from netforge_rl.datasets.minari_export import export_npz

    trajs = collect_dataset(
        lambda: make_policy(args.red),
        lambda: make_policy(args.policy),
        n_episodes=args.episodes,
        max_ticks=args.max_ticks,
        quality=args.quality,
    )
    if args.hdf5:
        from netforge_rl.datasets.minari_export import export_hdf5

        path = export_hdf5(trajs, args.out.with_suffix('.hdf5'))
    else:
        path = export_npz(trajs, args.out)
    print(f'wrote {len(trajs)} episodes -> {path}')
    return 0
