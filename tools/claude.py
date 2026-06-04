import anthropic


class ClaudeClient:
    def __init__(self, model_strong: str, model_fast: str):
        self.model_strong = model_strong
        self.model_fast = model_fast
        self._client = anthropic.AsyncAnthropic(max_retries=8, timeout=120.0)

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
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Concatenate all text blocks; tolerate an empty content list.
        return "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )
