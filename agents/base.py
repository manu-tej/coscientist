from __future__ import annotations
from pathlib import Path
from tools.claude import ClaudeClient


class BaseAgent:
    def __init__(self, client: ClaudeClient, prompts_dir: Path):
        self.client = client
        self.prompts_dir = Path(prompts_dir)

    def render_prompt(self, template_name: str, **variables) -> str:
        path = self.prompts_dir / f"{template_name}.txt"
        template = path.read_text()
        # Escape braces in user-supplied values to prevent format-string injection
        safe_variables = {
            k: str(v).replace("{", "{{").replace("}", "}}")
            for k, v in variables.items()
        }
        return template.format(**safe_variables)

    async def call_claude(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        use_strong: bool = False,
        max_tokens: int = 8192,
    ) -> str:
        return await self.client.call(
            system_prompt, user_prompt, use_strong=use_strong, max_tokens=max_tokens
        )

    async def run_turn_loop(
        self,
        template_name: str,
        variables: dict,
        transcript_key: str,
        termination_signal: str,
        max_turns: int = 10,
        system_prompt: str = "You are an expert scientific researcher.",
        use_strong: bool = False,
    ) -> tuple[str, list[str]]:
        transcript: list[str] = []
        last_response = ""
        for _ in range(max_turns):
            vars_with_transcript = {
                **variables,
                transcript_key: "\n".join(transcript),
            }
            prompt = self.render_prompt(template_name, **vars_with_transcript)
            response = await self.call_claude(
                system_prompt, prompt, use_strong=use_strong
            )
            transcript.append(response)
            last_response = response
            if termination_signal in response:
                break
        return last_response, transcript
