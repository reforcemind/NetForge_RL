from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResearchQuestion:
    id: str
    title: str
    question: str
    why: str
    protocol: str
    metrics: tuple[str, ...]
    status: str  # runnable | protocol
    note: str

    def as_dict(self) -> dict:
        return asdict(self)
