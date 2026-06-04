import os


class GoogleClient:
    """LLMClient implementation against the google-generativeai SDK (Gemini)."""

    def __init__(
        self,
        model='gemini-2.0-flash',
        *,
        api_key=None,
        max_tokens=256,
        system=None,
    ):
        try:
            import google.generativeai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                'google-generativeai SDK required. '
                'Install with `pip install google-generativeai`.'
            ) from e
        self.model_id = model
        self._key = api_key or os.environ.get('GOOGLE_API_KEY')
        if not self._key:
            raise ValueError('GOOGLE_API_KEY missing')
        self._max_tokens = max_tokens
        self._system = system or (
            'You are a network defense operator. Reply with exactly one '
            '`ACTION <id> TARGET <ip>` line and nothing else.'
        )
        self._model = None

    def _ensure(self):
        if self._model is None:
            import google.generativeai as genai

            genai.configure(api_key=self._key)
            self._model = genai.GenerativeModel(
                model_name=self.model_id,
                system_instruction=self._system,
            )
        return self._model

    def act(self, prompt):
        parts = [prompt['text']]
        if 'image_b64_png' in prompt:
            import base64

            parts.append(
                {
                    'mime_type': prompt.get('mime_type', 'image/png'),
                    'data': base64.b64decode(prompt['image_b64_png']),
                }
            )
        resp = self._ensure().generate_content(
            parts,
            generation_config={'max_output_tokens': self._max_tokens},
        )
        return getattr(resp, 'text', '') or ''
