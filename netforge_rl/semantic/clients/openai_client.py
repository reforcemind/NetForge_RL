import os

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class OpenAIClient:
    """LLMClient implementation against the OpenAI Chat Completions API."""

    def __init__(
        self,
        model='gpt-4o-mini',
        *,
        api_key=None,
        max_tokens=256,
        system=None,
    ):
        if not HAS_OPENAI:
            raise ImportError('openai SDK required.')
        self.model_id = model
        self._key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self._key:
            raise ValueError('OPENAI_API_KEY missing')
        self._max_tokens = max_tokens
        self._system = system or (
            'You are a network defense operator. Reply with exactly one '
            '`ACTION <id> TARGET <ip>` line and nothing else.'
        )
        self._client = None

    def _ensure(self):
        if self._client is None:
            self._client = openai.OpenAI(api_key=self._key)
        return self._client

    def act(self, prompt):
        content = [{'type': 'text', 'text': prompt['text']}]
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
