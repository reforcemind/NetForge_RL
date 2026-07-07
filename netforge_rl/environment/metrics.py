from netforge_rl.actions.attack_map import ALL_TECHNIQUE_IDS, technique_for
from netforge_rl.core.commands import iter_host_deltas
from netforge_rl.environment.constants import PADDING_SUBNET


class EpisodeMetricsMixin:
    """Episode security metrics and per-agent info extraction for NetForgeRLEnv."""

    def _active_hosts(self):
        """Real hosts, excluding inactive padding used only for fixed observation shape."""
        return [
            h
            for h in self.global_state.all_hosts.values()
            if h.subnet_cidr != PADDING_SUBNET
        ]

    def _update_episode_metrics(self, resolved_effects):
        for agent, effect in resolved_effects.items():
            if 'red' in agent:
                self.episode_metrics['red_actions'] += 1
                action = getattr(effect, 'action', None)
                technique = technique_for(type(action).__name__) if action else None
                if technique is not None:
                    self.episode_metrics['attack_techniques'].add(technique[0])
                target = self.global_state.all_hosts.get(
                    getattr(action, 'target_ip', None)
                )
                if target is not None and (
                    target.decoy not in (None, 'inactive')
                    or getattr(target, 'contains_honeytokens', False)
                ):
                    self.episode_metrics['deception_hits'] += 1
            if not effect.success:
                continue
            for attribute, ip, val in iter_host_deltas(effect.state_deltas):
                if attribute == 'privilege' and val in ('User', 'Root'):
                    self.episode_metrics['infection_times'].setdefault(
                        ip, self.current_tick
                    )
                elif attribute == 'status' and val == 'isolated':
                    self.episode_metrics['isolation_times'].setdefault(
                        ip, self.current_tick
                    )

        active = self._active_hosts()
        total = max(len(active), 1)
        healthy = sum(
            1 for h in active if h.compromised_by == 'None' and h.status == 'online'
        )
        self.episode_metrics['sla_uptime_sum'] += healthy / total
        self.episode_metrics['steps_count'] += 1

    def _extract_agent_infos(
        self, observations: dict, resolved_effects: dict, rewards: dict = None
    ) -> dict:
        infos = {}
        for agent in observations:
            agent_effect = resolved_effects.get(agent)
            info = self._action_outcome_info(agent_effect)
            self._add_host_metrics(info, agent)
            self._add_objective_metrics(info)
            raw_reward = (rewards or {}).get(agent, 0.0)
            info['normalized_reward'] = float(
                self.scenario.normalized_reward(raw_reward)
            )
            infos[agent] = info
        return infos

    def _action_outcome_info(self, agent_effect) -> dict:
        false_positives = successful_exploits = hosts_isolated = services_restored = 0
        if agent_effect and agent_effect.success:
            for attr, ip, val in iter_host_deltas(agent_effect.state_deltas):
                if attr == 'status' and val == 'isolated':
                    hosts_isolated += 1
                    host = self.global_state.all_hosts.get(ip)
                    if host and host.compromised_by == 'None':
                        false_positives += 1
                elif attr == 'privilege' and val in ('User', 'Root'):
                    successful_exploits += 1
                elif attr == 'status' and val == 'online':
                    services_restored += 1

        target_ip = (
            getattr(agent_effect.action, 'target_ip', None) if agent_effect else None
        )
        self.ordered_hosts = sorted(self.global_state.all_hosts.keys())
        return {
            'false_positives': float(false_positives),
            'successful_exploits': float(successful_exploits),
            'hosts_isolated': float(hosts_isolated),
            'services_restored': float(services_restored),
            'target_ip_index': (
                self.ordered_hosts.index(target_ip)
                if target_ip and target_ip in self.global_state.all_hosts
                else None
            ),
        }

    def _add_host_metrics(self, info: dict, agent: str) -> None:
        active = self._active_hosts()
        info['agent_energy'] = float(self.global_state.agent_energy.get(agent, 0))
        info['active_hosts'] = float(len(active))
        info['compromised_hosts'] = float(
            sum(1 for h in active if h.compromised_by != 'None')
        )
        info['isolated_hosts'] = float(sum(1 for h in active if h.status == 'isolated'))
        steps = self.episode_metrics['steps_count']
        info['SLA_Uptime_Percentage'] = float(
            self.episode_metrics['sla_uptime_sum'] / steps if steps > 0 else 1.0
        )
        info['MTTC'] = self._mean_time_to_containment()
        info['Total_Exfiltrated_Data'] = float(self.episode_metrics['exfiltrated_data'])

        deception_hits = self.episode_metrics['deception_hits']
        red_actions = self.episode_metrics['red_actions']
        info['deception_hits'] = float(deception_hits)
        info['deception_efficacy'] = float(
            deception_hits / red_actions if red_actions > 0 else 0.0
        )
        techniques = self.episode_metrics['attack_techniques']
        info['attack_techniques'] = sorted(techniques)
        info['attack_coverage'] = float(len(techniques) / len(ALL_TECHNIQUE_IDS))

    def _mean_time_to_containment(self) -> float:
        infections = self.episode_metrics['infection_times']
        vals = [
            t_iso - infections[ip]
            for ip, t_iso in self.episode_metrics['isolation_times'].items()
            if ip in infections
        ]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def _add_objective_metrics(self, info: dict) -> None:
        infections = self.episode_metrics['infection_times']
        all_compromised = set(infections.keys())
        detected = all_compromised & set(self.episode_metrics['isolation_times'].keys())
        info['detection_rate'] = float(len(detected) / max(len(all_compromised), 1))
        info['containment_time'] = float(
            sum(
                self.episode_metrics['isolation_times'][ip] - infections[ip]
                for ip in detected
            )
            / max(len(detected), 1)
        )

        active = self._active_hosts()
        total = max(len(active), 1)
        healthy = sum(
            1 for h in active if h.compromised_by == 'None' and h.status == 'online'
        )
        compromised = sum(1 for h in active if h.compromised_by != 'None')
        info['blue_score'] = float(
            (healthy / total) * 50.0
            + info['detection_rate'] * 30.0
            - info['containment_time'] * 0.1
        )
        info['red_score'] = float(
            (compromised / total) * 50.0
            + float(self.episode_metrics['exfiltrated_data']) * 5.0
        )
