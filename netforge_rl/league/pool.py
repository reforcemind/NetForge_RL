from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OpponentPool:
    """League of current, historical, and specialist opponent factories."""

    members: dict[str, Callable] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def add(self, name: str, factory: Callable) -> None:
        self.members[name] = factory
        self.history.append(name)

    def sample(self, rng: np.random.Generator | None = None) -> tuple[str, Callable]:
        rng = rng or np.random.default_rng()
        name = str(rng.choice(list(self.members)))
        return name, self.members[name]

    def historical(self, n: int = 3) -> list[tuple[str, Callable]]:
        names = self.history[-n:]
        return [(name, self.members[name]) for name in names if name in self.members]
