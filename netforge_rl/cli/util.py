def policies(blue_name: str, red_name: str):
    from netforge_rl.baselines.registry import make_policy

    return make_policy(red_name), make_policy(blue_name)
