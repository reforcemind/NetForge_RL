import random
from typing import Any, Dict


class GreenAgent:
    """Benign users producing background SIEM noise + false-positive alerts."""

    def __init__(self, agent_id: str = 'green_agent_0'):
        self.agent_id = agent_id
        from netforge_rl.siem.event_templates import evid_4624, evid_4688, sysmon_3

        self._benign_templates = [evid_4624, sysmon_3, evid_4688]

    def generate_noise(self, current_tick: int, global_state: Any) -> Dict[str, Any]:
        """Emit noise alerts based on a 150-tick day/night business cycle."""
        cycle_position = current_tick % 150
        is_day = cycle_position <= 100

        hosts = list(global_state.all_hosts.values())
        if not hosts:
            return {'alerts': []}

        noise_logs = []
        p_noise = 0.8 if is_day else 0.1
        p_fp = 0.05 if is_day else 0.01

        if random.random() < p_noise:
            source = random.choice(hosts)
            target = random.choice(hosts)
            if source.ip != target.ip:
                log = random.choice(self._benign_templates)(source.ip, target.ip)
                noise_logs.append(
                    {
                        'type': 'benign_xml',
                        'data': log,
                        'subnet': source.subnet_cidr,
                        'severity': 0,
                    }
                )

        if random.random() < p_fp:
            from netforge_rl.siem.event_templates import evid_4625

            target = random.choice(hosts)
            log = evid_4625('unknown_external', target.ip, username='Administrator')
            noise_logs.append(
                {
                    'type': 'anomaly_xml',
                    'data': log,
                    'subnet': target.subnet_cidr,
                    'severity': 3,
                }
            )

        return {'alerts': noise_logs}
