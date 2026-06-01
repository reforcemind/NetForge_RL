"""Anthropic / OpenAI client tests. Skipped if SDK not installed; uses
mocking to avoid network + key requirements.
"""

import sys
import types

import pytest


# ── Anthropic ──────────────────────────────────────────────────────────


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Inject a fake ``anthropic`` module with a recording client."""
    mod = types.ModuleType('anthropic')

    class FakeMsg:
        def __init__(self, text): self.content = [types.SimpleNamespace(type='text', text=text)]

    class FakeMessages:
        def __init__(self): self.last_kwargs = None
        def create(self, **kw):
            self.last_kwargs = kw
            return FakeMsg('ACTION 0 TARGET 10.0.0.1')

    class FakeAnthropic:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.messages = FakeMessages()

    mod.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, 'anthropic', mod)
    return mod


def test_anthropic_requires_api_key(fake_anthropic, monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    from netforge_rl.semantic.clients.anthropic_client import AnthropicClient
    with pytest.raises(ValueError, match='ANTHROPIC_API_KEY'):
        AnthropicClient()


def test_anthropic_act_text_only(fake_anthropic, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    from netforge_rl.semantic.clients.anthropic_client import AnthropicClient
    c = AnthropicClient(model='claude-sonnet-4-6')
    out = c.act({'text': 'hi'})
    assert out == 'ACTION 0 TARGET 10.0.0.1'
    kw = c._client.messages.last_kwargs
    assert kw['model'] == 'claude-sonnet-4-6'
    assert kw['messages'][0]['content'][0]['type'] == 'text'


def test_anthropic_act_with_image(fake_anthropic, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    from netforge_rl.semantic.clients.anthropic_client import AnthropicClient
    c = AnthropicClient()
    c.act({'text': 'hi', 'image_b64_png': 'AAA', 'mime_type': 'image/png'})
    types_ = [b['type'] for b in c._client.messages.last_kwargs['messages'][0]['content']]
    assert types_ == ['image', 'text']


# ── OpenAI ─────────────────────────────────────────────────────────────


@pytest.fixture
def fake_openai(monkeypatch):
    mod = types.ModuleType('openai')

    class FakeChoice:
        def __init__(self, text): self.message = types.SimpleNamespace(content=text)

    class FakeCompletions:
        def __init__(self): self.last_kwargs = None
        def create(self, **kw):
            self.last_kwargs = kw
            return types.SimpleNamespace(choices=[FakeChoice('ACTION 1 TARGET 10.0.0.2')])

    class FakeChat:
        def __init__(self): self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = FakeChat()

    mod.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, 'openai', mod)
    return mod


def test_openai_requires_api_key(fake_openai, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    from netforge_rl.semantic.clients.openai_client import OpenAIClient
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        OpenAIClient()


def test_openai_act_with_image(fake_openai, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    from netforge_rl.semantic.clients.openai_client import OpenAIClient
    c = OpenAIClient()
    out = c.act({'text': 'hi', 'image_b64_png': 'AAA', 'mime_type': 'image/png'})
    assert out == 'ACTION 1 TARGET 10.0.0.2'
    kw = c._client.chat.completions.last_kwargs
    user_content = kw['messages'][1]['content']
    assert any(b['type'] == 'image_url' for b in user_content)
    assert 'data:image/png;base64,AAA' in user_content[-1]['image_url']['url']
