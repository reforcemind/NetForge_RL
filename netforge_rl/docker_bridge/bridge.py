from __future__ import annotations
from netforge_rl.docker_bridge.docker_hypervisor import DockerHypervisor
from netforge_rl.docker_bridge.mock_hypervisor import MockHypervisor

import logging
from typing import Literal

from netforge_rl.docker_bridge.hypervisor_base import BaseHypervisor, HypervisorResult

logger = logging.getLogger(__name__)

_REWARD_DELTA: dict[str, float] = {
    'success': +5.0,
    'failure_clean': -10.0,
    'failure_noisy': -20.0,
    'failure_error': -25.0,
}

_NOISY_LATENCY_THRESHOLD_MS = 5000.0  # Longer than this = "noisy" failure


class DockerBridge:
    """Bridge connecting actions to the hypervisor backend."""

    def __init__(self, mode: Literal['sim', 'real'] = 'sim') -> None:
        self.mode = mode
        self._driver: BaseHypervisor = self._init_driver(mode)

    def dispatch(
        self,
        action_name: str,
        target_ip: str,
        target_os: str,
    ) -> HypervisorResult:
        """Execute payload; auto-fallback to mock if real driver is down."""
        result = self._driver.dispatch(action_name, target_ip, target_os)
        logger.debug('DockerBridge: %s', result)
        return result

    def reward_delta(self, result: HypervisorResult) -> float:
        """Map a HypervisorResult to an immediate scalar reward delta."""
        if result.success:
            return _REWARD_DELTA['success']
        elif result.return_code == 2:
            # Container/infrastructure error
            return _REWARD_DELTA['failure_error']
        elif result.latency_ms >= _NOISY_LATENCY_THRESHOLD_MS:
            return _REWARD_DELTA['failure_noisy']
        else:
            return _REWARD_DELTA['failure_clean']

    def teardown_all(self) -> None:
        """Destroy all active containers/sessions — call at episode end."""
        self._driver.teardown_all()

    def reseed(self, seed: int | None) -> None:
        """Reseed the driver RNG so exploit outcomes are reproducible per episode."""
        self._driver.reseed(seed)

    def is_available(self) -> bool:
        return self._driver.is_available()

    def _init_driver(self, mode: str) -> BaseHypervisor:
        if mode == 'real':
            driver = DockerHypervisor()
            if not driver.is_available():
                logger.warning(
                    'DockerBridge: real mode requested but Docker unavailable. '
                    'Falling back to mock hypervisor.'
                )
                return MockHypervisor()
            return driver

        # Default: sim / mock
        return MockHypervisor()
