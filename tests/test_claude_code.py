import os
import pytest
from tools import claude_code as cc
from tools.llm import LLMBackendError


class _FakeAssistant:
    def __init__(self, text):
        from claude_agent_sdk import TextBlock
        self.content = [TextBlock(text=text)] if text else []


def _make_result_message(result_text, is_error=False):
    """Build a real ResultMessage instance so the backend's isinstance check matches.

    ResultMessage is a non-frozen dataclass with several required fields; we
    construct it via __new__ and set only the attributes the backend reads.
    """
    from claude_agent_sdk import ResultMessage
    msg = ResultMessage.__new__(ResultMessage)
    msg.result = result_text
    msg.is_error = is_error
    return msg


def _make_fake_query(result_text, is_error=False, captured=None):
    async def fake_query(*, prompt, options):
        if captured is not None:
            captured["prompt"] = prompt
            captured["options"] = options
        yield _make_result_message(result_text, is_error=is_error)
    return fake_query


@pytest.mark.asyncio
async def test_call_returns_result_text(monkeypatch):
    captured = {}
    monkeypatch.setattr(cc, "query", _make_fake_query("hello world", captured=captured))
    backend = cc.ClaudeCodeBackend(model_strong="opus", model_fast="sonnet")
    out = await backend.call("sys", "user", use_strong=False)
    assert out == "hello world"
    # fast model selected when use_strong=False
    assert captured["options"].model == "sonnet"
    assert captured["options"].tools == []
    assert captured["options"].max_turns == 1


@pytest.mark.asyncio
async def test_call_uses_strong_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(cc, "query", _make_fake_query("x", captured=captured))
    backend = cc.ClaudeCodeBackend()
    await backend.call("sys", "user", use_strong=True)
    assert captured["options"].model == "opus"


@pytest.mark.asyncio
async def test_call_strips_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-should-be-removed")
    captured = {}
    monkeypatch.setattr(cc, "query", _make_fake_query("x", captured=captured))
    await cc.ClaudeCodeBackend().call("sys", "user")
    assert "ANTHROPIC_API_KEY" not in captured["options"].env


@pytest.mark.asyncio
async def test_call_raises_on_error_result(monkeypatch):
    monkeypatch.setattr(cc, "query", _make_fake_query("rate limited", is_error=True))
    with pytest.raises(LLMBackendError):
        await cc.ClaudeCodeBackend().call("sys", "user")
