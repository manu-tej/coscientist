import os
import pytest
from tools.llm import make_backend, LLMBackendError


_CFG = {
    "provider": "claude-code",
    "anthropic": {"model_strong": "claude-opus-4-7", "model_fast": "claude-sonnet-4-6"},
    "claude_code": {"model_strong": "opus", "model_fast": "sonnet"},
    "codex": {
        "model": "codex-test-model",
        "effort_strong": "high",
        "effort_fast": "low",
    },
    "openai_compatible": {
        "base_url": "https://llm.example/v1",
        "api_key_env": "TEST_LLM_API_KEY",
        "model_strong": "large-model",
        "model_fast": "small-model",
        "timeout_seconds": 12,
    },
}


def test_make_backend_claude_code_default(monkeypatch):
    monkeypatch.delenv("COSCIENTIST_PROVIDER", raising=False)
    from tools.claude_code import ClaudeCodeBackend
    b = make_backend(_CFG)
    assert isinstance(b, ClaudeCodeBackend)
    assert b.model_strong == "opus" and b.model_fast == "sonnet"


def test_make_backend_anthropic(monkeypatch):
    monkeypatch.delenv("COSCIENTIST_PROVIDER", raising=False)
    from tools.claude import ClaudeClient
    cfg = {**_CFG, "provider": "anthropic"}
    b = make_backend(cfg)
    assert isinstance(b, ClaudeClient)


def test_env_override_beats_cfg(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_PROVIDER", "anthropic")
    from tools.claude import ClaudeClient
    b = make_backend(_CFG)   # cfg says claude-code, env says anthropic
    assert isinstance(b, ClaudeClient)


def test_explicit_provider_override_beats_environment(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_PROVIDER", "anthropic")
    from tools.claude_code import ClaudeCodeBackend
    b = make_backend(_CFG, provider_override="claude-code")
    assert isinstance(b, ClaudeCodeBackend)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_PROVIDER", "bogus")
    with pytest.raises(LLMBackendError):
        make_backend(_CFG)


def test_make_backend_codex(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_PROVIDER", "codex")
    from tools.codex import CodexBackend
    backend = make_backend(_CFG)
    assert isinstance(backend, CodexBackend)
    assert backend.model == "codex-test-model"
    assert backend.model_strong == "codex-test-model"
    assert backend.model_fast == "codex-test-model"
    assert backend.effort_strong == "high"
    assert backend.effort_fast == "low"


def test_make_backend_openai_compatible(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_PROVIDER", "openai-compatible")
    from tools.openai_compatible import OpenAICompatibleBackend
    backend = make_backend(_CFG)
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.base_url == "https://llm.example/v1"
    assert backend.api_key_env == "TEST_LLM_API_KEY"
    assert backend.model_strong == "large-model"
    assert backend.model_fast == "small-model"
    assert backend.timeout_seconds == 12


def test_make_backend_default_provider_when_missing(monkeypatch):
    monkeypatch.delenv("COSCIENTIST_PROVIDER", raising=False)
    from tools.claude_code import ClaudeCodeBackend
    b = make_backend({"claude_code": {}})   # no provider key → default claude-code
    assert isinstance(b, ClaudeCodeBackend)
