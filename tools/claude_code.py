from __future__ import annotations

import os

from claude_agent_sdk import (
    query, ClaudeAgentOptions, AssistantMessage, TextBlock, ResultMessage,
)
from tools.llm import LLMBackendError


class ClaudeCodeBackend:
    """LLM backend that runs completions on a Claude Code / Max subscription via
    the claude-agent-sdk, instead of the metered Anthropic API. One turn, no tools."""

    def __init__(self, model_strong: str = "opus", model_fast: str = "sonnet"):
        self.model_strong = model_strong
        self.model_fast = model_fast

    def _clean_env(self) -> dict:
        # Strip ANTHROPIC_API_KEY so the SDK uses subscription OAuth, not per-token billing.
        return {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    async def call(self, system_prompt, user_prompt, *, use_strong=False, max_tokens=8192) -> str:
        model = self.model_strong if use_strong else self.model_fast
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=model,
            tools=[],
            max_turns=1,
            setting_sources=None,
            env=self._clean_env(),
        )
        text_parts: list[str] = []
        final = None
        is_error = False
        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                final = message.result
                is_error = bool(getattr(message, "is_error", False))
        if is_error:
            raise LLMBackendError(f"Claude Code backend returned an error result: {final!r}")
        return final if final is not None else "".join(text_parts)
