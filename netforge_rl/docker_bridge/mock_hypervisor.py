from __future__ import annotations

import json
import random
import time
from pathlib import Path

from netforge_rl.docker_bridge.hypervisor_base import BaseHypervisor, HypervisorResult

# Base success probabilities.
_DEFAULT_SUCCESS_RATES: dict[str, float] = {
    'ExploitEternalBlue': 0.72,
    'ExploitBlueKeep': 0.58,
    'ExploitHTTP_RFI': 0.65,
    'ExploitRemoteService': 0.55,
    'PrivilegeEscalate': 0.70,
    'DumpLSASS': 0.80,
    'PassTheTicket': 0.90,
}

_OS_PENALTY: dict[str, float] = {
    # OS penalty offsets.
    'Linux_Ubuntu': {'ExploitEternalBlue': -0.60, 'ExploitBlueKeep': -0.70},
    'Linux_CentOS': {'ExploitEternalBlue': -0.60, 'ExploitBlueKeep': -0.70},
    'PLC_Firmware': {
        'ExploitEternalBlue': -0.90,
        'ExploitBlueKeep': -0.90,
        'ExploitHTTP_RFI': -0.90,
    },
}

# Latency distribution (mean, std).
_LATENCY_PROFILE: dict[str, tuple[float, float]] = {
    'ExploitEternalBlue': (4200.0, 800.0),
    'ExploitBlueKeep': (3800.0, 600.0),
    'ExploitHTTP_RFI': (1200.0, 300.0),
    'ExploitRemoteService': (2500.0, 500.0),
    'PrivilegeEscalate': (1800.0, 400.0),
    'DumpLSASS': (900.0, 200.0),
    'PassTheTicket': (600.0, 150.0),
}
_DEFAULT_LATENCY = (2000.0, 600.0)


class MockHypervisor(BaseHypervisor):
    """Zero-dependency mock hypervisor for fast execution."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        library_path = Path(__file__).parent / 'payload_library.json'
        with open(library_path) as f:
            self._library: dict = json.load(f)

    def reseed(self, seed: int | None) -> None:
        self._rng = random.Random(seed)

    def dispatch(
        self,
        action_name: str,
        target_ip: str,
        target_os: str,
    ) -> HypervisorResult:
        """Synthesise a realistic payload result."""
        t_start = time.perf_counter()

        success = self._roll_success(action_name, target_os)
        stdout = self._sample_stdout(action_name, success, target_ip)
        return_code = 0 if success else 1

        mean, std = _LATENCY_PROFILE.get(action_name, _DEFAULT_LATENCY)
        latency_ms = max(50.0, self._rng.gauss(mean, std))

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        latency_ms = max(latency_ms, elapsed_ms)

        return HypervisorResult(
            success=success,
            stdout=stdout,
            return_code=return_code,
            latency_ms=round(latency_ms, 1),
            action_name=action_name,
            target_ip=target_ip,
            target_os=target_os,
        )

    def teardown_all(self) -> None:
        """No-op."""
        pass

    def is_available(self) -> bool:
        return True

    def _roll_success(self, action_name: str, target_os: str) -> bool:
        base_rate = _DEFAULT_SUCCESS_RATES.get(action_name, 0.50)
        penalty = _OS_PENALTY.get(target_os, {}).get(action_name, 0.0)
        adjusted = max(0.02, min(0.98, base_rate + penalty))
        return self._rng.random() < adjusted

    def _sample_stdout(self, action_name: str, success: bool, target_ip: str) -> str:
        bucket = self._library.get(action_name)
        if bucket is None:
            if success:
                return f'[*] {action_name} succeeded against {target_ip}\n[*] Session opened.'
            return (
                f'[-] {action_name} failed against {target_ip}\n[-] No session created.'
            )

        key = 'success' if success else 'failure'
        samples = bucket.get(key, [])
        if not samples:
            return f'[*] {action_name} {"completed" if success else "failed"}.'

        template = self._rng.choice(samples)

        return template.replace('10.0.1.3', target_ip).replace('10.0.0.7', target_ip)
