from netforge_rl.arena.spec import ARENA_NAME


def cmd_version(_args=None) -> int:
    try:
        from importlib.metadata import PackageNotFoundError, version

        ver = version('netforge-rl')
    except PackageNotFoundError:
        from netforge_rl import __version__ as ver
    print(f'netforge-rl {ver}')
    print(ARENA_NAME)
    return 0
