from __future__ import annotations

from pathlib import Path
from typing import Callable
from tools.llm import LLMBackend
from tools.resources import resource_path


class AgentOutputError(RuntimeError):
    """A model response could not satisfy an agent's portable output contract."""


# Optional markdown decoration models wrap around field labels (**Label:**,
# ## Label:, __Label__:) even when prompts require exact plain labels — observed
# live on 2026-08-04, where it failed every generation task of a run. Parsers
# interpolate this around labels so formatting cannot fail a run while missing
# content still does.
LABEL_DECO = r"[ \t]*[#*_]*[ \t]*"


class BaseAgent:
    def __init__(self, client: LLMBackend, prompts_dir: Path):
        self.client = client
        self.prompts_dir = resource_path(prompts_dir)

    def render_prompt(self, template_name: str, **variables) -> str:
        path = self.prompts_dir / f"{template_name}.txt"
        template = path.read_text()
        # ``str.format`` substitutes values verbatim; it does not parse braces
        # inside replacement values.  This matters for structured JSON context
        # passed between the company hierarchy and its specialist agents.
        return template.format(**{key: str(value) for key, value in variables.items()})

    async def call_claude(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        use_strong: bool = False,
        max_tokens: int = 8192,
    ) -> str:
        response = await self.client.call(
            system_prompt, user_prompt, use_strong=use_strong, max_tokens=max_tokens
        )
        if not isinstance(response, str) or not response.strip():
            raise AgentOutputError("LLM backend returned an empty response")
        return response

    async def run_turn_loop(
        self,
        template_name: str,
        variables: dict,
        transcript_key: str,
        termination_signal: str,
        max_turns: int = 10,
        system_prompt: str = "You are an expert scientific researcher.",
        use_strong: bool = False,
        is_complete: "Callable[[str], bool] | None" = None,
    ) -> tuple[str, list[str]]:
        """Run a multi-turn loop until the model signals completion.

        The signal alone is a weak test: the prompts contain the signal word, so
        models echo it while still deliberating ("we will converge on a HYPOTHESIS
        after more debate"). Ending there yields an unfinished turn that fails the
        caller's parser, discarding the whole discussion. ``is_complete`` lets a
        caller require that the turn actually satisfies its output contract, giving
        the model further turns to comply instead of failing the task.
        """
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
            if termination_signal in response and (
                is_complete is None or is_complete(response)
            ):
                break
        return last_response, transcript
