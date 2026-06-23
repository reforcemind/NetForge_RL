import random
from typing import Any, Optional
from netforge_rl.actions.blue.mitigation import IsolateHost, RestoreHost
from netforge_rl.actions.blue.analysis import Analyze, Monitor
from netforge_rl.actions.blue.deception import DeployHoneytoken
from netforge_rl.actions.blue.edr import DeployEDR


class RuleBasedBlueAgent:
    """Deterministic Blue defender for use as a lower-bound paper baseline."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._isolated: set = set()
        self._honeytokened: set = set()
        self._edr_deployed: set = set()

    def reset(self) -> None:
        self._isolated.clear()
        self._honeytokened.clear()
        self._edr_deployed.clear()

    def get_action(self, _observation: Any, global_state) -> Optional[Any]:
        live = {
            ip: h
            for ip, h in global_state.all_hosts.items()
            if h.status == 'online' and '169.254' not in ip
        }

        for ip, host in live.items():
            if host.privilege in ('User', 'Root') and ip not in self._isolated:
                self._isolated.add(ip)
                return IsolateHost(self.agent_id, ip)

        for ip, host in live.items():
            if host.edr_active and ip not in self._isolated:
                return Analyze(self.agent_id, ip)

        candidates = sorted(
            [
                h
                for ip, h in live.items()
                if not h.edr_active and ip not in self._edr_deployed
            ],
            key=lambda h: getattr(h, 'cvss_score', 0.0),
            reverse=True,
        )
        if candidates:
            target = candidates[0]
            self._edr_deployed.add(target.ip)
            return DeployEDR(self.agent_id, target.ip)

        candidates = sorted(
            [
                h
                for ip, h in live.items()
                if not h.contains_honeytokens and ip not in self._honeytokened
            ],
            key=lambda h: getattr(h, 'cvss_score', 0.0),
            reverse=True,
        )
        if candidates:
            target = candidates[0]
            self._honeytokened.add(target.ip)
            return DeployHoneytoken(self.agent_id, target.ip)

        for ip, host in global_state.all_hosts.items():
            if host.status == 'isolated' and host.compromised_by == 'None':
                return RestoreHost(self.agent_id, ip)

        if live:
            target_ip = random.choice(list(live.keys()))
            return Monitor(self.agent_id, target_ip)

        return None
