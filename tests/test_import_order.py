import subprocess
import sys

import pytest

# baselines.eval and netforge_rl.semantic reference each other; importing either
# first must not deadlock the module system. These run in fresh interpreters so a
# cycle cannot be masked by import order elsewhere in the test session.
ORDERS = [
    'import netforge_rl.semantic.clients.mock; import netforge_rl.baselines',
    'import netforge_rl.baselines; import netforge_rl.semantic',
    'from netforge_rl.baselines import evaluate; import netforge_rl.semantic',
]


@pytest.mark.fast
@pytest.mark.parametrize('snippet', ORDERS)
def test_no_circular_import(snippet):
    proc = subprocess.run(
        [sys.executable, '-c', snippet + '; print("ok")'],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'ok' in proc.stdout
