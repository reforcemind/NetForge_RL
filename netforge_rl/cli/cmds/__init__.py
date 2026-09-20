from netforge_rl.cli.cmds.arena import cmd_arena
from netforge_rl.cli.cmds.benchmark import cmd_benchmark
from netforge_rl.cli.cmds.collect import cmd_collect
from netforge_rl.cli.cmds.evaluate import cmd_evaluate
from netforge_rl.cli.cmds.questions import cmd_questions
from netforge_rl.cli.cmds.replay import cmd_replay
from netforge_rl.cli.cmds.run import cmd_run
from netforge_rl.cli.cmds.test_policy import cmd_test_policy
from netforge_rl.cli.cmds.version import cmd_version

COMMANDS = {
    'run': cmd_run,
    'test-policy': cmd_test_policy,
    'benchmark': cmd_benchmark,
    'arena': cmd_arena,
    'replay': cmd_replay,
    'collect': cmd_collect,
    'questions': cmd_questions,
    'evaluate': cmd_evaluate,
    'version': cmd_version,
}
