# Pluggable LLM Backends — Design Specification

**Date:** 2026-06-01
**Goal:** Let the AI co-scientist framework be powered by a Claude Code / Claude Max subscription or an OpenAI Codex / ChatGPT subscription, instead of only the pay-per-token Anthropic API — so the user can run the framework on subscriptions they already pay for.

---

## 1. Goals & Scope

Introduce a pluggable LLM-backend abstraction with three interchangeable implementations selected by config:

- **anthropic** — existing pay-per-token Anthropic API (unchanged behavior).
- **claude-code** — the user's Claude Code / Max subscription via the `claude-agent-sdk` Python package (**new default**).
- **codex** — the user's OpenAI Codex / ChatGPT subscription via the `codex exec` CLI in non-interactive mode.

**In scope:**
- One provider per run, selectable via `config.yaml` (`provider:`) and an env override (`COSCIENTIST_PROVIDER`).
- The existing `use_strong` flag maps to a strong/fast distinction *within* each provider.
- Auth hygiene enforced in code (strip the conflicting API-key env var so subscription billing is used, not per-token).
- Mocked unit tests for both new backends and the factory — no real subscriptions or API keys required to run the suite.

**Out of scope (v1):**
- Per-agent provider routing (assigning a different backend to each of the 6 agents).
- Mixing providers within a run (e.g. strong→Claude Code, fast→Codex). The `make_backend` factory is structured so this can be added later without touching agent code.
- A UI control for provider selection (config/env only).

---

## 2. Why this is a small change

Every LLM call in the codebase already funnels through a single seam:

```
agent code → BaseAgent.call_claude(system, user, use_strong=...) → self.client.call(system, user, use_strong=..., max_tokens=...) → str
```

`BaseAgent` and `ConfigParser` are duck-typed on a `client` object exposing `.call(...)`. Therefore new backends are drop-in: implement the same `.call()` signature and inject the chosen one at the entrypoint. No agent, orchestrator, supervisor, or pipeline code changes.

---

## 3. The Backend Contract

`tools/llm.py` defines the contract and the factory.

```python
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
    """Raised when a backend fails (subprocess error, rate limit, missing CLI)."""


def make_backend(cfg: dict) -> LLMBackend:
    """Construct the backend named by cfg['provider'] (env COSCIENTIST_PROVIDER
    overrides). Raises LLMBackendError on an unknown provider name."""
```

`make_backend` reads `provider = os.environ.get("COSCIENTIST_PROVIDER") or cfg.get("provider", "claude-code")` and returns:
- `"anthropic"` → `AnthropicBackend(model_strong=cfg["anthropic"]["model_strong"], model_fast=cfg["anthropic"]["model_fast"])`
- `"claude-code"` → `ClaudeCodeBackend(model_strong=cfg["claude_code"]["model_strong"], model_fast=cfg["claude_code"]["model_fast"])`
- `"codex"` → `CodexBackend(model=cfg["codex"]["model"], effort_strong=cfg["codex"]["effort_strong"], effort_fast=cfg["codex"]["effort_fast"])`
- anything else → raise `LLMBackendError(f"Unknown provider: {provider!r}")`

**Circular-import avoidance:** `LLMBackendError` and the `LLMBackend` Protocol live in `tools/llm.py`. The backend modules (`claude_code.py`, `codex.py`) import `LLMBackendError` from `tools/llm.py` at module top. To avoid a cycle, `make_backend` imports the three backend classes **lazily inside the function body** (not at module top):

```python
def make_backend(cfg: dict) -> "LLMBackend":
    import os
    provider = os.environ.get("COSCIENTIST_PROVIDER") or cfg.get("provider", "claude-code")
    if provider == "anthropic":
        from tools.claude import ClaudeClient
        return ClaudeClient(model_strong=cfg["anthropic"]["model_strong"],
                            model_fast=cfg["anthropic"]["model_fast"])
    if provider == "claude-code":
        from tools.claude_code import ClaudeCodeBackend
        return ClaudeCodeBackend(model_strong=cfg["claude_code"]["model_strong"],
                                 model_fast=cfg["claude_code"]["model_fast"])
    if provider == "codex":
        from tools.codex import CodexBackend
        return CodexBackend(model=cfg["codex"]["model"],
                            effort_strong=cfg["codex"]["effort_strong"],
                            effort_fast=cfg["codex"]["effort_fast"])
    raise LLMBackendError(f"Unknown provider: {provider!r}")
```

(The `AnthropicBackend = ClaudeClient` alias in §4 is for documentation/readability; the factory imports `ClaudeClient` directly to keep the lazy import simple.)

---

## 4. AnthropicBackend (existing — unchanged)

`tools/claude.py` keeps the current `ClaudeClient` class exactly as-is (constructor `ClaudeClient(model_strong, model_fast)`, `async call(...)` using `AsyncAnthropic` with prompt caching). For naming clarity `tools/llm.py` imports it and exposes the alias `AnthropicBackend = ClaudeClient`. No behavioral change; the existing `tests/test_claude.py` stays green untouched.

---

## 5. ClaudeCodeBackend (new — subscription via Agent SDK)

`tools/claude_code.py`.

```python
import os
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock, ResultMessage
from tools.llm import LLMBackendError


class ClaudeCodeBackend:
    def __init__(self, model_strong: str = "opus", model_fast: str = "sonnet"):
        self.model_strong = model_strong
        self.model_fast = model_fast

    def _clean_env(self) -> dict:
        # Strip ANTHROPIC_API_KEY so the SDK uses subscription OAuth, not per-token billing.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        return env

    async def call(self, system_prompt, user_prompt, *, use_strong=False, max_tokens=8192) -> str:
        model = self.model_strong if use_strong else self.model_fast
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,   # plain string => replaces Claude Code's default agent prompt
            model=model,                   # "opus" | "sonnet"
            tools=[],                      # disable all built-in tools -> pure completion
            max_turns=1,                   # single turn, no agentic loop
            setting_sources=None,          # do not load ~/.claude or repo CLAUDE.md/hooks/MCP
            env=self._clean_env(),
        )
        text_parts: list[str] = []
        final: str | None = None
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
```

**Design notes:**
- `system_prompt` as a plain string fully replaces Claude Code's built-in agent prompt — required for a clean completion (a preset would only append).
- `tools=[]` + `max_turns=1` guarantees no tool use / no agentic loop.
- `setting_sources=None` prevents the spawned CLI from loading the host repo's `CLAUDE.md`, hooks, or MCP servers (which would pollute prompts and add latency).
- Text is taken from `ResultMessage.result` (the consolidated final text), falling back to concatenated `TextBlock`s.
- `is_error` covers rate-limit exhaustion — relevant because, as of **2026-06-15**, subscription Agent-SDK usage draws from a separate monthly "Agent SDK credit" pool. The Supervisor already catches per-task exceptions, so a raised `LLMBackendError` degrades gracefully (that task is logged and skipped, the run continues).
- `max_tokens` is accepted for interface parity; the Agent SDK manages output length itself, so it is not forwarded.

**SDK version caveat (verify at implementation time):** the `ClaudeAgentOptions` field names (`tools`, `setting_sources`, `env`, `system_prompt`, `model`, `max_turns`) and the message/result types are accurate per the 2026-06 research, but `claude-agent-sdk` evolves. The implementer must `pip show claude-agent-sdk` and confirm the field names against the installed version before finalizing — the unit tests construct real `ClaudeAgentOptions`, so a renamed field would surface at test time. The exact model-id strings for `model=` should be the aliases `"opus"`/`"sonnet"` (resolved by the CLI), not pinned full IDs.

**Auth setup (documented for the user, not code):**
- Interactive: run `claude` once and `/login`. Ensure `ANTHROPIC_API_KEY` is unset.
- Headless/server: `claude setup-token` → export `CLAUDE_CODE_OAUTH_TOKEN`.

---

## 6. CodexBackend (new — subscription via `codex exec`)

`tools/codex.py`.

```python
import asyncio
import os
import shutil
import tempfile
from tools.llm import LLMBackendError


class CodexBackend:
    def __init__(self, model: str = "gpt-5.1-codex", effort_strong: str = "high", effort_fast: str = "medium"):
        self.model = model
        self.effort_strong = effort_strong
        self.effort_fast = effort_fast

    def _clean_env(self) -> dict:
        # Strip OPENAI_API_KEY so Codex uses ChatGPT subscription auth, not API billing.
        return {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}

    async def call(self, system_prompt, user_prompt, *, use_strong=False, max_tokens=8192) -> str:
        if shutil.which("codex") is None:
            raise LLMBackendError("codex CLI not found. Install with: npm i -g @openai/codex")
        effort = self.effort_strong if use_strong else self.effort_fast
        # No --system flag exists; prepend the system prompt into the message (most reliable).
        combined = f"{system_prompt}\n\n---\n\n{user_prompt}"
        scratch = tempfile.mkdtemp(prefix="coscientist_codex_")
        try:
            proc = await asyncio.create_subprocess_exec(
                "codex", "exec",
                "-c", f"model={self.model}",
                "-c", f"model_reasoning_effort={effort}",
                "-c", "tools.web_search=disabled",
                "-c", "features.shell_tool=false",
                "-c", "hide_agent_reasoning=true",
                "--sandbox", "read-only",
                "-a", "never",
                "--skip-git-repo-check",
                "--ephemeral",
                "--cd", scratch,
                combined,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._clean_env(),
            )
            out, err = await proc.communicate()
            if proc.returncode != 0:
                raise LLMBackendError(f"codex exec failed (exit {proc.returncode}): {err.decode()[:500]}")
            return out.decode().strip()
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
```

**Design notes:**
- `codex exec` prints only the final agent message to stdout; progress goes to stderr — so plain stdout capture is the clean text path.
- Tool-neutralization: `--sandbox read-only` + `-a never` + `features.shell_tool=false` + `tools.web_search=disabled` prevents file edits, shell, and web use, so it behaves as a text completer.
- `--ephemeral` + a per-call `--cd` scratch dir prevents session/rollout files and cross-worker collisions; the scratch dir is removed in `finally`.
- `use_strong` maps to `model_reasoning_effort` (high vs medium).
- Missing-binary check gives an actionable install message.
- `max_tokens` accepted for parity; Codex manages its own output length.

**Auth setup (documented for the user, not code):**
- `codex login` once → cached in `~/.codex/auth.json`. Ensure `OPENAI_API_KEY` is unset.

---

## 7. Config Changes

`config.yaml` gains a `provider` key (default `claude-code`) and two provider blocks. Existing `anthropic:` block is retained for the anthropic provider.

```yaml
provider: claude-code          # claude-code | anthropic | codex
                               # env override: COSCIENTIST_PROVIDER

anthropic:                     # used when provider: anthropic
  model_strong: claude-opus-4-7
  model_fast: claude-sonnet-4-6

claude_code:                   # used when provider: claude-code
  model_strong: opus
  model_fast: sonnet

codex:                         # used when provider: codex
  model: gpt-5.1-codex
  effort_strong: high
  effort_fast: medium
```

---

## 8. Entrypoint Wiring

In `coscientist.py`, replace the direct `ClaudeClient(...)` construction with the factory:

```python
from tools.llm import make_backend
...
backend = make_backend(cfg)
base = BaseAgent(client=backend, prompts_dir=prompts_dir)
parser = ConfigParser(client=backend, prompts_dir=prompts_dir)
```

`RankingAgent`, `ProximityAgent`, `SearchTool`, etc. are unaffected. The same `backend` instance powers config parsing, safety review, and all agents.

---

## 9. Dependencies

- Add `claude-agent-sdk>=0.1.0` to `pyproject.toml` dependencies (needed for the default provider).
- The `codex` CLI is a **runtime** dependency only when `provider: codex`; it is not a pip dependency. `CodexBackend.call` raises an actionable error if the binary is absent.

---

## 10. Error Handling Summary

| Failure | Backend | Behavior |
|---|---|---|
| Unknown provider name | factory | `make_backend` raises `LLMBackendError` at startup |
| ANTHROPIC_API_KEY set (would override sub) | claude-code | stripped from subprocess env automatically |
| OPENAI_API_KEY set (would override sub) | codex | stripped from subprocess env automatically |
| Agent SDK credit / rate limit | claude-code | `ResultMessage.is_error` → `LLMBackendError` → Supervisor logs + skips task |
| `codex` binary missing | codex | `LLMBackendError` with `npm i -g @openai/codex` hint |
| `codex exec` nonzero exit | codex | `LLMBackendError` with stderr excerpt |

Because the Supervisor wraps every `runner.run_task` in try/except (errors logged, worker survives), a backend error degrades to a skipped task rather than crashing the run.

---

## 11. Testing Strategy

All tests mock the transport — no subscriptions, API keys, or network needed.

- `tests/test_llm.py` — `make_backend` returns the correct type for each provider string; `COSCIENTIST_PROVIDER` env overrides `cfg["provider"]`; unknown provider raises `LLMBackendError`.
- `tests/test_claude_code.py` — monkeypatch `claude_agent_sdk.query` (async generator) to yield a fake `ResultMessage(result="...", is_error=False)`; assert `.call` returns the text, that `ClaudeAgentOptions` was built with `tools=[]`, `max_turns=1`, and `model=opus` when `use_strong=True` / `sonnet` otherwise; assert `is_error=True` raises `LLMBackendError`; assert `ANTHROPIC_API_KEY` is absent from the options env.
- `tests/test_codex.py` — mock `asyncio.create_subprocess_exec` to return a fake process with canned stdout and returncode 0; assert the argv contains `--sandbox read-only`, `-a never`, `features.shell_tool=false`, the correct `model_reasoning_effort` per `use_strong`, and that the system prompt is prepended into the final positional arg; assert returned text equals stdout; assert nonzero returncode raises `LLMBackendError`; assert `OPENAI_API_KEY` is stripped from the passed env; mock `shutil.which` to simulate missing binary → raises.
- Existing 98 tests remain unchanged and green (the AnthropicBackend path is the same `ClaudeClient` they already cover).

---

## 12. Verification (post-implementation)

Beyond the mocked unit tests, a runtime check (only runnable where the user is logged in):
- With `provider: claude-code` and a valid `CLAUDE_CODE_OAUTH_TOKEN` (and `ANTHROPIC_API_KEY` unset), run a short `coscientist.py` goal and confirm hypotheses are produced through the subscription — observable in the DB, no API key consumed.
- This is a manual/optional step, gated on the user's subscription login, not part of the automated suite.
