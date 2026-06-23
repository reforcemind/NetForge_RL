import random
from typing import Any, Dict, List, Tuple


_TEMP_TAU = 20.0
_PRESSURE_TAU = 10.0
_FLOW_TAU = 5.0

_TEMP_ALARM = 80.0
_TEMP_CRITICAL = 120.0
_PRESSURE_ALARM_HIGH = 130.0
_PRESSURE_ALARM_LOW = 70.0
_PRESSURE_CRITICAL_HIGH = 180.0
_PRESSURE_CRITICAL_LOW = 30.0
_FLOW_ALARM_HIGH = 90.0
_FLOW_ALARM_LOW = 20.0
_FLOW_CRITICAL_HIGH = 150.0
_FLOW_CRITICAL_LOW = 5.0


class PLCPhysicsEngine:
    """First-order lag ODE physics for OT/SCADA PLC hosts."""

    def __init__(self):
        self._rng = random.Random()

    def reset(self, seed: int = None) -> None:
        self._rng = random.Random(seed)

    def tick(self, global_state) -> Tuple[List[Dict[str, Any]], List[Tuple[str, Any]]]:
        """Evolve physical state for all PLC hosts.

        Returns (siem_alerts, state_deltas). No-op if no PLC hosts exist.
        """
        alerts: List[Dict[str, Any]] = []
        deltas: List[Tuple[str, Any]] = []

        for ip, host in global_state.all_hosts.items():
            if getattr(host, 'os', None) != 'PLC_Firmware':
                continue
            if host.status == 'isolated':
                continue
            if getattr(host, 'system_integrity', 'clean') == 'kinetic_destruction':
                continue

            temp = getattr(host, 'temperature', 50.0)
            pressure = getattr(host, 'pressure', 100.0)
            flow = getattr(host, 'flow_rate', 50.0)
            temp_sp = getattr(host, 'temperature_setpoint', temp)
            pressure_sp = getattr(host, 'pressure_setpoint', pressure)
            flow_sp = getattr(host, 'flow_rate_setpoint', flow)

            temp += (temp_sp - temp) / _TEMP_TAU + self._rng.gauss(0, 0.1)
            pressure += (pressure_sp - pressure) / _PRESSURE_TAU + self._rng.gauss(
                0, 0.2
            )
            flow += (flow_sp - flow) / _FLOW_TAU + self._rng.gauss(0, 0.3)

            deltas.extend(
                [
                    (f'hosts/{ip}/temperature', round(temp, 2)),
                    (f'hosts/{ip}/pressure', round(pressure, 2)),
                    (f'hosts/{ip}/flow_rate', round(flow, 2)),
                ]
            )

            kinetic = (
                temp > _TEMP_CRITICAL
                or pressure > _PRESSURE_CRITICAL_HIGH
                or pressure < _PRESSURE_CRITICAL_LOW
                or flow > _FLOW_CRITICAL_HIGH
                or flow < _FLOW_CRITICAL_LOW
            )
            alarm = not kinetic and (
                temp > _TEMP_ALARM
                or pressure > _PRESSURE_ALARM_HIGH
                or pressure < _PRESSURE_ALARM_LOW
                or flow > _FLOW_ALARM_HIGH
                or flow < _FLOW_ALARM_LOW
            )

            if kinetic:
                deltas.append((f'hosts/{ip}/system_integrity', 'kinetic_destruction'))
                alerts.append(
                    {
                        'signature': 'SCADA_KINETIC_BREACH',
                        'target': ip,
                        'temperature': round(temp, 1),
                        'pressure': round(pressure, 1),
                        'flow_rate': round(flow, 1),
                        'severity': 10,
                    }
                )
            elif alarm:
                alerts.append(
                    {
                        'signature': 'SCADA_PHYSICAL_ALARM',
                        'target': ip,
                        'temperature': round(temp, 1),
                        'pressure': round(pressure, 1),
                        'flow_rate': round(flow, 1),
                        'severity': 7,
                    }
                )

        return alerts, deltas
