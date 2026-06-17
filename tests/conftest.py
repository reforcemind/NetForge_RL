import pytest
from netforge_rl.environment.parallel_env import NetForgeRLEnv
from netforge_rl.docker_bridge.bridge import DockerBridge
from netforge_rl.siem.siem_logger import SIEMLogger
from netforge_rl.nlp.log_encoder import LogEncoder

@pytest.fixture
def env_config():
    return {'scenario_type': 'ransomware', 'docker_bridge_mode': 'sim', 'nlp_backend': 'tfidf', 'max_ticks': 100, 'log_latency': 2}

@pytest.fixture
def env_sim(env_config):
    env = NetForgeRLEnv(env_config)
    env.reset(seed=42)
    return env

@pytest.fixture
def global_state():
    from netforge_rl.topologies.network_generator import NetworkGenerator
    gen = NetworkGenerator()
    state = gen.generate(seed=0)
    return state

@pytest.fixture
def mock_bridge():
    return DockerBridge(mode='sim')

@pytest.fixture
def siem_logger():
    return SIEMLogger(seed=0)

@pytest.fixture
def log_encoder():
    return LogEncoder(backend='tfidf')

@pytest.fixture
def red_agent_id():
    return 'red_operator_0'

@pytest.fixture
def blue_agent_id():
    return 'blue_operator_0'