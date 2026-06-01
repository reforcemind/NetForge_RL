"""Provider-neutral LLM client interface."""

from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    """Minimal interface every provider adapter implements.

    ``prompt`` is the dict produced by :func:`build_vla_prompt`
    (``{'text', 'image_b64_png', 'mime_type'}``); ``image_b64_png`` is
    optional — text-only clients ignore it.

    ``model_id`` identifies the underlying model for leaderboard logging.
    """

    model_id: str

    def act(self, prompt: dict[str, Any]) -> str:
        """Return the model's reply as plain text."""
        ...
