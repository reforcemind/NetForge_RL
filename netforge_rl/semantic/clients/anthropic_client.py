"""Anthropic API client for the zero-shot leaderboard.

Lazy import of the ``anthropic`` SDK so the package stays light when
this client isn't used. Reads ``ANTHROPIC_API_KEY`` from the environment.
"""

from __future__ import annotations

import os
from typing import Any


class AnthropicClient:
    def __init__(
        self,
        model: str = 'claude-sonnet-4-6',
        *,
        api_key: str | None = None,
        max_tokens: int = 256,
        system: str | None = None,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "anthropic SDK required. Install with `pip install anthropic`."
            ) from e
        self.model_id = model
        self._key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self._key:
            raise ValueError('ANTHROPIC_API_KEY missing')
        self._max_tokens = max_tokens
        self._system = (
            system
            or 'You are a network defense operator. Reply with exactly one '
            '`ACTION <id> TARGET <ip>` line and nothing else.'
        )
        self._client = None

    def _ensure(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._key)
        return self._client

    def act(self, prompt: dict[str, Any]) -> str:
        content: list[dict] = [{'type': 'text', 'text': prompt['text']}]
        if 'image_b64_png' in prompt:
            content.insert(
                0,
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': prompt.get('mime_type', 'image/png'),
                        'data': prompt['image_b64_png'],
                    },
                },
            )
        msg = self._ensure().messages.create(
            model=self.model_id,
            max_tokens=self._max_tokens,
            system=self._system,
            messages=[{'role': 'user', 'content': content}],
        )
        return ''.join(b.text for b in msg.content if getattr(b, 'type', '') == 'text')
