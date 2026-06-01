"""Scenario registry. Add new scenarios with :func:`register`.

Built-ins are registered lazily on first lookup so importing this package
stays cheap (and the heavy scenario classes only pull in their own deps
when actually needed).
"""

from __future__ import annotations

from typing import Callable, Type

from netforge_rl.scenarios.base_scenario import BaseScenario


_REGISTRY: dict[str, Callable[[], Type[BaseScenario]]] = {}


def register(name: str, loader: Callable[[], Type[BaseScenario]]) -> None:
    """Register a scenario class loader under ``name`` (case-insensitive)."""
    _REGISTRY[name.lower()] = loader


def get_scenario_class(name: str) -> Type[BaseScenario]:
    """Resolve a scenario name to its class, loading on first request.

    Raises :class:`KeyError` with the list of known scenarios on miss.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f'Unknown scenario {name!r}. Available: {sorted(_REGISTRY)}'
        )
    return _REGISTRY[key]()


def available() -> list[str]:
    return sorted(_REGISTRY)


def _ransomware():
    from netforge_rl.scenarios.ransomware import RansomwareScenario
    return RansomwareScenario


def _apt():
    from netforge_rl.scenarios.apt_espionage import AptEspionageScenario
    return AptEspionageScenario


def _iot_grid():
    from netforge_rl.scenarios.iot_grid import IoTGridScenario
    return IoTGridScenario


def _ot_stuxnet():
    from netforge_rl.scenarios.ot_stuxnet import OTStuxnetScenario
    return OTStuxnetScenario


register('ransomware', _ransomware)
register('apt_espionage', _apt)
register('iot_grid', _iot_grid)
register('ot_stuxnet', _ot_stuxnet)
