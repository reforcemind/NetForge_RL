from __future__ import annotations

import json
from pathlib import Path


def cmd_replay(args) -> int:
    from netforge_rl.render.replay_html import write_replay_html

    payload = json.loads(Path(args.trajectory).read_text())
    out = write_replay_html(payload, args.out)
    print(out)
    return 0
