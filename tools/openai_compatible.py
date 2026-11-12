from __future__ import annotations

import os
from typing import Any

import httpx

from tools.llm import LLMBackendError


class OpenAICompatibleBackend:
    """Chat-completions backend for OpenAI-compatible hosted or local models."""

    def __init__(
        self,
        *,
        base_url: str,
        model_strong: str,
        model_fast: str,
        api_key_env: str | None = "OPENAI_API_KEY",
        timeout_seconds: float = 60.0,
    ):
        if not base_url:
            raise LLMBackendError(
                "openai-compatible provider requires a non-empty base_url"
            )
        self.base_url = base_url.rstrip("/")
        self.model_strong = model_strong
        self.model_fast = model_fast
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        # The factory shares one backend across all agents. Lazily reusing one
        # async client gives concurrent workers a bounded connection pool and
        # avoids a TLS handshake for every model call.
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
        return self._client

    async def aclose(self) -> None:
        """Close pooled HTTP connections when an embedding application shuts down."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise LLMBackendError(
                    f"{self.api_key_env} is not set — required for the "
                    "'openai-compatible' provider"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def _extract_text(payload: Any) -> str:
        try:
            choice = payload["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMBackendError(
                "OpenAI-compatible response did not contain choices[0]"
            ) from exc

        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "".join(parts)
        # Some compatible servers expose the older completions-style `text`.
        if isinstance(choice, dict) and isinstance(choice.get("text"), str):
            return choice["text"]
        raise LLMBackendError(
            "OpenAI-compatible response did not contain text content"
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
        request = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
        }
        headers = self._headers()
        try:
            response = await self._get_client().post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()[:500]
            raise LLMBackendError(
                "OpenAI-compatible request failed "
                f"(HTTP {exc.response.status_code}): {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMBackendError(
                f"OpenAI-compatible request failed: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMBackendError(
                "OpenAI-compatible response was not valid JSON"
            ) from exc
        return self._extract_text(payload)
