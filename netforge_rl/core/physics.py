from typing import Dict

from netforge_rl.core.action import ActionEffect


def _targets(eff: ActionEffect) -> set:
    """IPs touched by an effect — works for both dict-form and command-form deltas."""
    if eff is None or not eff.success:
        return set()
    if isinstance(eff.state_deltas, dict):
        return {
            k.split('/')[1]
            for k in eff.state_deltas
            if isinstance(k, str) and 'hosts/' in k
        }
    if isinstance(eff.state_deltas, list):
        return {
            d.target_ip
            for d in eff.state_deltas
            if getattr(d, 'target_ip', None)
        }
    return set()


class ConflictResolutionEngine:
    """Blue defensive supremacy on same-target collision in the same tick."""

    @staticmethod
    def resolve(effects: Dict[str, ActionEffect]) -> Dict[str, ActionEffect]:
        defended = set()
        for agent_id, eff in effects.items():
            if 'blue' in agent_id.lower():
                defended |= _targets(eff)

        for agent_id, eff in effects.items():
            if 'red' not in agent_id.lower() or eff is None or not eff.success:
                continue
            if _targets(eff) & defended:
                eff.success = False
                eff.state_deltas = [] if isinstance(eff.state_deltas, list) else {}
                eff.observation_data['alert'] = 'TEMPORAL_COLLISION_DEFENSE_SUPREMACY'

        return effects
