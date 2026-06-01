"""LLM client adapters for the zero-shot leaderboard."""

from netforge_rl.semantic.clients.base import LLMClient
from netforge_rl.semantic.clients.mock import MockLLMClient

__all__ = ['LLMClient', 'MockLLMClient']
