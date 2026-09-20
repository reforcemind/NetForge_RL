from netforge_rl.cli.util import policies


def cmd_run(args) -> int:
    from netforge_rl.arena.evaluate import run_episode
    from netforge_rl.render.replay_html import ReplayLog, write_replay_html
    from netforge_rl.scenarios.yaml_dsl import make_env_from_spec, resolve_scenario

    spec = resolve_scenario(args.scenario)
    red, blue = policies(args.policy, args.red)
    rec = run_episode(
        red,
        blue,
        scenario=args.scenario,
        seed=args.seed,
        max_ticks=args.max_ticks,
    )
    print(f'{spec.name} seed={args.seed}')
    m = rec.metrics
    print(
        f'  mission={m.mission_success:.0f}  SLA={m.sla_uptime:.2f}  '
        f'security={m.security:.2f}  FP={m.false_positives:.0f}  '
        f'catastrophic={m.catastrophic_failure:.0f}'
    )
    if args.replay:
        env = make_env_from_spec(spec, seed=None, max_ticks=args.max_ticks)
        log = ReplayLog(scenario=spec.name, seed=args.seed)
        env.reset(seed=args.seed)
        red, blue = policies(args.policy, args.red)
        while env.agents:
            actions = {}
            for agent_id in env.agents:
                pol = red if 'red' in agent_id else blue
                actions[agent_id] = pol.act(env, agent_id)
            _, rewards, term, trunc, infos = env.step(actions)
            log.capture(env, rewards, infos, actions)
            if all(term.values()) or all(trunc.values()):
                break
        html = (
            args.replay.with_suffix('.html')
            if args.replay.suffix != '.html'
            else args.replay
        )
        write_replay_html(log.to_dict(), html)
        log.save_json(html.with_suffix('.json'))
        print(f'  replay: {html}')
    return 0
