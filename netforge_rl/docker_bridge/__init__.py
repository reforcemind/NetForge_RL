"""
NetForge_RL docker_bridge Package.

Provides a dual-mode hypervisor bridge for connecting the MARL environment
to either a lightweight MockHypervisor (for fast RL training) or a live
DockerHypervisor (for high-fidelity evaluation runs).
"""

from netforge_rl.docker_bridge.hypervisor_base import BaseHypervisor, HypervisorResult
from netforge_rl.docker_bridge.mock_hypervisor import MockHypervisor
from netforge_rl.docker_bridge.bridge import DockerBridge

__all__ = [
    'BaseHypervisor',
    'HypervisorResult',
    'MockHypervisor',
    'DockerBridge',
]
