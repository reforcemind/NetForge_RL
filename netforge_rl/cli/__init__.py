"""NetForge CLI."""

from __future__ import annotations

import sys

from netforge_rl.cli.cmds import COMMANDS
from netforge_rl.cli.parser import build_parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
