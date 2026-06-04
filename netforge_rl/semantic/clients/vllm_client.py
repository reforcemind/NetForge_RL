import os


class VLLMClient:
    """LLMClient against a local vLLM server's OpenAI-compatible endpoint.

    Use it the same way as OpenAIClient but pointed at your own vLLM
    deployment — defaults assume `vllm serve` running on
    http://localhost:8000/v1.
    """

    def __init__(
        self,
        model='meta-llama/Meta-Llama-3-8B-Instruct',
        *,
        base_url=None,
        api_key=None,
        max_tokens=256,
        system=None,
    ):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                'openai SDK required (vLLM exposes the OpenAI-compatible API). '
                'Install with `pip install openai`.'
            ) from e
        self.model_id = model
        self._base_url = base_url or os.environ.get(
            'VLLM_BASE_URL', 'http://localhost:8000/v1'
        )
        self._key = api_key or os.environ.get('VLLM_API_KEY', 'EMPTY')
        self._max_tokens = max_tokens
        self._system = system or (
            'You are a network defense operator. Reply with exactly one '
            '`ACTION <id> TARGET <ip>` line and nothing else.'
        )
        self._client = None

    def _ensure(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self._key, base_url=self._base_url)
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
