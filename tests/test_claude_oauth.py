"""ClaudeOAuthBackend system-block construction.

Regression: an empty system_prompt must NOT produce a second system block with
cache_control — the API rejects "cache_control cannot be set for empty text
blocks" with a 400. The base-model reference baseline calls with a minimal
system prompt, but the backend must be robust to an empty one too.
"""
import asyncio
import types

import pytest


def _make_backend(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    captured = {}

    class FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            block = types.SimpleNamespace(type="text", text="ok")
            return types.SimpleNamespace(content=[block])

    class FakeClient:
        def __init__(self, *a, **k):
            self.messages = FakeMessages()

    import anthropic
    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeClient)
    from tools.claude_oauth import ClaudeOAuthBackend
    return ClaudeOAuthBackend("strong-model", "fast-model"), captured


def test_empty_system_prompt_omits_second_block(monkeypatch):
    backend, captured = _make_backend(monkeypatch)
    asyncio.run(backend.call("", "hi", use_strong=False))
    system = captured["system"]
    assert len(system) == 1  # only the identity block
    assert "cache_control" not in system[0]


def test_nonempty_system_prompt_adds_cached_block(monkeypatch):
    backend, captured = _make_backend(monkeypatch)
    asyncio.run(backend.call("You are helpful.", "hi", use_strong=True))
    system = captured["system"]
    assert len(system) == 2
    assert system[1]["text"] == "You are helpful."
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    assert captured["model"] == "strong-model"
