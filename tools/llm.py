from __future__ import annotations

import os
from typing import Protocol


class LLMBackend(Protocol):
    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        use_strong: bool = False,
        max_tokens: int = 8192,
    ) -> str:
        ...


class LLMBackendError(RuntimeError):
    """Raised when a backend fails (subprocess error, rate limit, missing CLI, unknown provider)."""


def make_backend(cfg: dict) -> "LLMBackend":
    """Construct the backend named by cfg['provider']; env COSCIENTIST_PROVIDER overrides.
    Default provider is 'claude-code' (the subscription path). Lazy imports avoid cycles."""
    provider = os.environ.get("COSCIENTIST_PROVIDER") or cfg.get("provider", "claude-code")
    if provider == "anthropic":
        from tools.claude import ClaudeClient
        return ClaudeClient(model_strong=cfg["anthropic"]["model_strong"],
                            model_fast=cfg["anthropic"]["model_fast"])
    if provider == "claude-code":
        from tools.claude_code import ClaudeCodeBackend
        cc = cfg.get("claude_code", {})
        return ClaudeCodeBackend(model_strong=cc.get("model_strong", "opus"),
                                 model_fast=cc.get("model_fast", "sonnet"))
    if provider == "codex":
        raise LLMBackendError("provider 'codex' is designed but not yet implemented in v1; use 'claude-code' or 'anthropic'.")
    raise LLMBackendError(f"Unknown provider: {provider!r}")
