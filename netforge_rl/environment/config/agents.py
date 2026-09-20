from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentIds:
    red: str = 'red_operator'
    blue_dmz: str = 'blue_dmz'
    blue_internal: str = 'blue_internal'
    blue_restricted: str = 'blue_restricted'

    def all(self) -> tuple[str, ...]:
        return (self.red, self.blue_dmz, self.blue_internal, self.blue_restricted)


AGENTS = AgentIds()
POSSIBLE_AGENTS: tuple[str, ...] = AGENTS.all()
