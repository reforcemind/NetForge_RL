from .core.action import ActionEffect, BaseAction
from .core.observation import BaseObservation
from .core.state import GlobalNetworkState, Host, Subnet
from .environment.parallel_env import NetForgeRLEnv

__version__ = '4.0.0'

__all__ = [
    'ActionEffect',
    'BaseAction',
    'BaseObservation',
    'GlobalNetworkState',
    'Host',
    'NetForgeRLEnv',
    'Subnet',
    '__version__',
]


def _register_third_party_envs() -> None:
    from netforge_rl.environment.registry import register_envs

    register_envs()


_register_third_party_envs()
