from typing import Any
import random
import numpy as np


class BLineAgent:
    """
    Scripted Red Agent that executes the exact B-Line killchain:

    DiscoverRemoteSystems -> DiscoverNetworkServices -> ExploitRemoteService -> PrivilegeEscalate -> Impact
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.known_hosts = []
        self.exploited_hosts = []
        self.root_hosts = []
        self.impacted_hosts = []
        self.step_count = 0

    def _pick_and_track(self, candidates_iter, tracking_list, ActionClass):
        candidates = list(candidates_iter)
        if candidates:
            target = random.choice(candidates)
            tracking_list.append(target)
            return ActionClass(self.agent_id, target)
        return None

    def get_action(self, _observation: np.ndarray, global_state) -> Any:
        from netforge_rl.actions import (
            DiscoverRemoteSystems,
            DiscoverNetworkServices,
            ExploitRemoteService,
            PrivilegeEscalate,
            Impact,
        )

        self.step_count += 1

        if not self.known_hosts or self.step_count < 3:
            available_subnets = list(global_state.subnets.keys())
            target_subnet = available_subnets[
                self.step_count % len(available_subnets)
            ] if available_subnets else '10.0.0.0/24'
            for host in global_state.all_hosts.values():
                if (
                    host.subnet_cidr == target_subnet
                    and host.ip not in self.known_hosts
                ):
                    self.known_hosts.append(host.ip)
            return DiscoverRemoteSystems(self.agent_id, target_subnet)

        action = self._pick_and_track(
            (h for h in self.known_hosts if h not in self.exploited_hosts and global_state.can_route_to(h)),
            self.exploited_hosts,
            ExploitRemoteService
        )
        if action: return action

        action = self._pick_and_track(
            (h for h in self.exploited_hosts if h not in self.root_hosts),
            self.root_hosts,
            PrivilegeEscalate
        )
        if action: return action

        action = self._pick_and_track(
            (h for h in self.root_hosts if h not in self.impacted_hosts),
            self.impacted_hosts,
            Impact
        )
        if action: return action

        target = random.choice(self.known_hosts) if self.known_hosts else '127.0.0.1'
        return DiscoverNetworkServices(self.agent_id, target)
