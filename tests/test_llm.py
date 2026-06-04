import os
import pytest
from tools.llm import make_backend, LLMBackendError


_CFG = {
    "provider": "claude-code",
    "anthropic": {"model_strong": "claude-opus-4-7", "model_fast": "claude-sonnet-4-6"},
    "claude_code": {"model_strong": "opus", "model_fast": "sonnet"},
}


def test_make_backend_claude_code_default():
    from tools.claude_code import ClaudeCodeBackend
    b = make_backend(_CFG)
    assert isinstance(b, ClaudeCodeBackend)
    assert b.model_strong == "opus" and b.model_fast == "sonnet"


def test_make_backend_anthropic():
    from tools.claude import ClaudeClient
    cfg = {**_CFG, "provider": "anthropic"}
    b = make_backend(cfg)
    assert isinstance(b, ClaudeClient)


def test_env_override_beats_cfg(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_PROVIDER", "anthropic")
    from tools.claude import ClaudeClient
    b = make_backend(_CFG)   # cfg says claude-code, env says anthropic
    assert isinstance(b, ClaudeClient)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_PROVIDER", "bogus")
    with pytest.raises(LLMBackendError):
        make_backend(_CFG)


def test_codex_not_implemented(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_PROVIDER", "codex")
    with pytest.raises(LLMBackendError):
        make_backend(_CFG)


def test_make_backend_default_provider_when_missing(monkeypatch):
    monkeypatch.delenv("COSCIENTIST_PROVIDER", raising=False)
    from tools.claude_code import ClaudeCodeBackend
    b = make_backend({"claude_code": {}})   # no provider key → default claude-code
    assert isinstance(b, ClaudeCodeBackend)
