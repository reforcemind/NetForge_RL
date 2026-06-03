from typing import Protocol


class LLMClient(Protocol):
    """Provider-neutral LLM client interface."""

    model_id: str

    def act(self, prompt: dict) -> str: ...
