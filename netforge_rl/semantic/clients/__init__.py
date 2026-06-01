"""LLM client adapters for the zero-shot leaderboard.

Real adapters (Anthropic, OpenAI) are not eagerly imported — they require
optional SDKs and an API key. Import them directly when you need them.
"""

from netforge_rl.semantic.clients.base import LLMClient
from netforge_rl.semantic.clients.mock import MockLLMClient

__all__ = ['LLMClient', 'MockLLMClient']
