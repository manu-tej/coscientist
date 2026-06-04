from __future__ import annotations

import os

import anthropic

from tools.llm import LLMBackendError

# Subscription OAuth tokens used against the Messages API require:
#   1. Bearer auth (the SDK's auth_token=), not x-api-key.
#   2. the oauth beta header.
#   3. the first system block to be the Claude Code identity, or the API 401/403s.
_CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
_OAUTH_BETA = "oauth-2025-04-20"


class ClaudeOAuthBackend:
    """LLM backend that runs on a Claude subscription via an OAuth token used with
    the standard Anthropic API client (Bearer auth), instead of a metered API key
    or the claude-agent-sdk. The token is read from an env var (never hardcoded)."""

    def __init__(
        self,
        model_strong: str,
        model_fast: str,
        token_env: str = "ANTHROPIC_AUTH_TOKEN",
    ):
        token = os.environ.get(token_env)
        if not token:
            raise LLMBackendError(
                f"{token_env} is not set — required for the 'claude-oauth' provider. "
                f"Put it in .env or run `claude setup-token`."
            )
        self.model_strong = model_strong
        self.model_fast = model_fast
        self._client = anthropic.AsyncAnthropic(
            auth_token=token,
            default_headers={"anthropic-beta": _OAUTH_BETA},
        )

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        use_strong: bool = False,
        max_tokens: int = 8192,
    ) -> str:
        model = self.model_strong if use_strong else self.model_fast
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[
                {"type": "text", "text": _CLAUDE_CODE_IDENTITY},
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
