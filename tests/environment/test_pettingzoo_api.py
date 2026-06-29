import pytest

pytest.importorskip('pettingzoo')
from pettingzoo.test import parallel_api_test

from netforge_rl.environment.parallel_env import NetForgeRLEnv


@pytest.mark.integration
def test_parallel_api_conformance() -> None:
    env = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 50})
    parallel_api_test(env, num_cycles=200)
