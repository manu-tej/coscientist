from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

from tools.llm import LLMBackendError


class CodexBackend:
    """Run single-turn completions through a logged-in Codex CLI subscription."""

    def __init__(
        self,
        model: str = "gpt-5.1-codex",
        model_strong: str | None = None,
        model_fast: str | None = None,
        effort_strong: str = "high",
        effort_fast: str = "medium",
    ):
        # ``model`` is retained for config/backward compatibility. Supplying
        # strong/fast IDs makes this backend obey the same routing contract as
        # every hosted or local backend.
        self.model = model
        self.model_strong = model_strong or model
        self.model_fast = model_fast or model
        self.effort_strong = effort_strong
        self.effort_fast = effort_fast

    def _clean_env(self) -> dict[str, str]:
        # A configured API key can make the CLI use metered API billing instead
        # of the user's cached ChatGPT subscription login.
        return {key: value for key, value in os.environ.items() if key != "OPENAI_API_KEY"}

    @staticmethod
    async def _terminate_process(proc) -> None:
        """Stop a cancelled CLI child so benchmark cancellation leaves no orphan."""
        if proc.returncode is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                return
            await proc.wait()

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        use_strong: bool = False,
        max_tokens: int = 8192,
    ) -> str:
        """Return the final Codex message.

        Codex does not expose a stable output-token flag, so ``max_tokens`` is
        accepted for parity with the common backend protocol but is not passed
        to the CLI.
        """
        if shutil.which("codex") is None:
            raise LLMBackendError(
                "codex CLI not found. Install with: npm i -g @openai/codex"
            )

        effort = self.effort_strong if use_strong else self.effort_fast
        model = self.model_strong if use_strong else self.model_fast
        # `codex exec` has no system-prompt flag. Keeping the boundary explicit
        # preserves the two-part prompt contract used by all co-scientist agents.
        combined_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        scratch = tempfile.mkdtemp(prefix="coscientist_codex_")
        try:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "codex",
                    # Approval policy belongs to the top-level CLI, not `exec`.
                    "-a",
                    "never",
                    "exec",
                    "-c",
                    f"model={model}",
                    "-c",
                    f"model_reasoning_effort={effort}",
                    "-c",
                    "tools.web_search=disabled",
                    "-c",
                    "features.shell_tool=false",
                    "-c",
                    "hide_agent_reasoning=true",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--cd",
                    scratch,
                    combined_prompt,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self._clean_env(),
                )
            except OSError as exc:
                raise LLMBackendError(f"failed to start codex CLI: {exc}") from exc

            try:
                stdout, stderr = await proc.communicate()
            except asyncio.CancelledError:
                await self._terminate_process(proc)
                raise
            if proc.returncode != 0:
                detail = stderr.decode(errors="replace").strip()[:500]
                raise LLMBackendError(
                    f"codex exec failed (exit {proc.returncode}): {detail}"
                )
            return stdout.decode(errors="replace").strip()
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
