from netforge_rl.baselines.policies import HeuristicBluePolicy, KillChainRedPolicy


def blue() -> HeuristicBluePolicy:
    return HeuristicBluePolicy(seed=0)


def killchain() -> KillChainRedPolicy:
    return KillChainRedPolicy(seed=0)
