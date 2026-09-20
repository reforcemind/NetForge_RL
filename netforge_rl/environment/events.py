from __future__ import annotations

from dataclasses import dataclass

from netforge_rl.core.action import ActionEffect, BaseAction


@dataclass
class PendingAction:
    completion_tick: int
    agent: str
    action: BaseAction
    effect: ActionEffect
    target_ip: str | None
    start_tick: int = 0

    def __getitem__(self, key: str):
        return getattr(self, key)


def as_pending(event) -> PendingAction:
    if isinstance(event, PendingAction):
        return event
    return PendingAction(
        completion_tick=int(event['completion_tick']),
        agent=event['agent'],
        action=event['action'],
        effect=event['effect'],
        target_ip=event.get('target_ip'),
        start_tick=int(event.get('start_tick', 0)),
    )
