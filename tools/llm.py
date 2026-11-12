from __future__ import annotations

import os
import inspect
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


def load_backend_environment() -> None:
    """Load local backend credentials/provider selection when dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def resolve_provider(cfg: dict, provider_override: str | None = None) -> str:
    """Resolve a provider without silently overriding an explicit caller choice.

    Interactive entrypoints retain the convenient ``COSCIENTIST_PROVIDER``
    override. Benchmark cells pass ``provider_override`` so the model recorded in
    a sweep is necessarily the model that was executed.
    """
    return (
        provider_override
        or os.environ.get("COSCIENTIST_PROVIDER")
        or cfg.get("provider", "claude-code")
    )


_PROVIDER_SECTIONS = {
    "anthropic": "anthropic",
    "claude-code": "claude_code",
    "codex": "codex",
    "openai-compatible": "openai_compatible",
    "openai_compatible": "openai_compatible",
    "offline": "offline",
}

# Public, canonical provider names for CLI `choices=`. The underscore spelling of
# openai_compatible is an accepted alias, not a separate provider, so it is excluded.
PROVIDERS = frozenset(_PROVIDER_SECTIONS) - {"openai_compatible"}


def configure_backend(
    cfg: dict,
    *,
    provider: str | None = None,
    model_strong: str | None = None,
    model_fast: str | None = None,
) -> tuple[str, dict]:
    """Apply exact provider/model overrides to a loaded configuration."""
    selected = resolve_provider(cfg, provider)
    section = _PROVIDER_SECTIONS.get(selected, selected.replace("-", "_"))
    if section not in cfg:
        raise LLMBackendError(
            f"provider {selected!r} has no configuration section {section!r}"
        )
    model_cfg = dict(cfg[section])
    if model_strong is not None:
        model_cfg["model_strong"] = model_strong
    if model_fast is not None:
        model_cfg["model_fast"] = model_fast
    if selected == "codex":
        legacy = model_cfg.get("model", "gpt-5.1-codex")
        model_cfg.setdefault("model_strong", legacy)
        model_cfg.setdefault("model_fast", legacy)
    cfg["provider"] = selected
    cfg[section] = model_cfg
    return selected, model_cfg


def make_backend(
    cfg: dict, *, provider_override: str | None = None,
) -> "LLMBackend":
    """Construct a configured backend.

    ``provider_override`` has highest precedence, followed by
    ``COSCIENTIST_PROVIDER`` and then ``cfg['provider']``. Lazy imports avoid
    optional-provider import cycles.
    """
    provider = resolve_provider(cfg, provider_override)
    if provider == "offline":
        from tools.offline import OfflineBackend
        off = cfg.get("offline", {})
        return OfflineBackend(seed=off.get("seed", 0))
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
        from tools.codex import CodexBackend
        codex = cfg.get("codex", {})
        legacy_model = codex.get("model", "gpt-5.1-codex")
        return CodexBackend(
            model=legacy_model,
            model_strong=codex.get("model_strong", legacy_model),
            model_fast=codex.get("model_fast", legacy_model),
            effort_strong=codex.get("effort_strong", "high"),
            effort_fast=codex.get("effort_fast", "medium"),
        )
    if provider in {"openai-compatible", "openai_compatible"}:
        from tools.openai_compatible import OpenAICompatibleBackend
        compatible = cfg.get("openai_compatible", {})
        return OpenAICompatibleBackend(
            base_url=compatible.get("base_url", "https://api.openai.com/v1"),
            api_key_env=compatible.get("api_key_env", "OPENAI_API_KEY"),
            model_strong=compatible.get("model_strong", "gpt-5"),
            model_fast=compatible.get("model_fast", "gpt-5-mini"),
            timeout_seconds=compatible.get("timeout_seconds", 60.0),
        )
    raise LLMBackendError(f"Unknown provider: {provider!r}")


async def close_backend(backend: object) -> None:
    """Close a backend-owned connection pool when it exposes a close hook."""
    closer = getattr(backend, "aclose", None) or getattr(backend, "close", None)
    if closer is None:
        return
    result = closer()
    if inspect.isawaitable(result):
        await result
