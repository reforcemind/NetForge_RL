from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class HypervisorResult:
    """Raw outcome of a dispatched payload."""

    success: bool
    stdout: str
    return_code: int
    latency_ms: float
    action_name: str
    target_ip: str
    target_os: str

    def __repr__(self) -> str:
        status = 'SUCCESS' if self.success else 'FAILED'
        return (
            f'<HypervisorResult [{status}] {self.action_name} → {self.target_ip} '
            f'({self.target_os}) | RC={self.return_code} | {self.latency_ms:.1f}ms>'
        )


class BaseHypervisor(ABC):
    """Abstract hypervisor driver interface."""

    @abstractmethod
    def dispatch(
        self,
        action_name: str,
        target_ip: str,
        target_os: str,
    ) -> HypervisorResult:
        """Dispatch a named payload action against a target host."""
        ...

    @abstractmethod
    def teardown_all(self) -> None:
        """Destroy all active execution contexts."""
        ...

    def is_available(self) -> bool:
        """Indicate whether this driver is operational."""
        return True
