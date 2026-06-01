"""OpenAI API client (chat.completions) for the zero-shot leaderboard.

Lazy import; reads ``OPENAI_API_KEY``. Supports text + image (data URI)
content blocks.
"""

from __future__ import annotations

import os
from typing import Any


class OpenAIClient:
    def __init__(
        self,
        model: str = 'gpt-4o-mini',
        *,
        api_key: str | None = None,
        max_tokens: int = 256,
        system: str | None = None,
    ):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "openai SDK required. Install with `pip install openai`."
            ) from e
        self.model_id = model
        self._key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self._key:
            raise ValueError('OPENAI_API_KEY missing')
        self._max_tokens = max_tokens
        self._system = (
            system
            or 'You are a network defense operator. Reply with exactly one '
            '`ACTION <id> TARGET <ip>` line and nothing else.'
        )
        self._client = None

    def _ensure(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self._key)
        return self._client

    def act(self, prompt: dict[str, Any]) -> str:
        content: list[dict] = [{'type': 'text', 'text': prompt['text']}]
        if 'image_b64_png' in prompt:
            mime = prompt.get('mime_type', 'image/png')
            content.append(
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:{mime};base64,{prompt["image_b64_png"]}',
                    },
                }
            )
        resp = self._ensure().chat.completions.create(
            model=self.model_id,
            max_tokens=self._max_tokens,
            messages=[
                {'role': 'system', 'content': self._system},
                {'role': 'user', 'content': content},
            ],
        )
        return resp.choices[0].message.content or ''
