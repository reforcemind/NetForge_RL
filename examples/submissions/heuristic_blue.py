"""CAGE-style ``make_blue`` example.

netforge evaluate examples/submissions/heuristic_blue.py --seeds 0 --max-ticks 20
"""

NAME = 'heuristic-blue'
TEAM = 'netforge-baselines'
TECHNIQUE = 'scripted-siem-isolate'
PAPER = ''


def wrap(env):
    return env


def make_blue(seed=0):
    from netforge_rl.baselines.policies import HeuristicBluePolicy

    return HeuristicBluePolicy(seed=seed)
