import asyncio
import shutil
import subprocess

import pytest

from tools import codex
from tools.llm import LLMBackendError


class _FakeProcess:
    def __init__(self, stdout=b"final answer\n", stderr=b"", returncode=0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self):
        return self._stdout, self._stderr


def _install_fake_process(monkeypatch, process=None):
    captured = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return process or _FakeProcess()

    monkeypatch.setattr(codex.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(
        codex.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("use_strong", "expected_effort"),
    [(False, "low"), (True, "high")],
)
async def test_call_builds_tool_neutral_command(
    monkeypatch, use_strong, expected_effort
):
    captured = _install_fake_process(monkeypatch)
    backend = codex.CodexBackend(
        model="test-model", effort_strong="high", effort_fast="low"
    )

    result = await backend.call(
        "system instructions", "user question", use_strong=use_strong
    )

    assert result == "final answer"
    args = captured["args"]
    assert args[:4] == ("codex", "-a", "never", "exec")
    assert "model=test-model" in args
    assert f"model_reasoning_effort={expected_effort}" in args
    assert "tools.web_search=disabled" in args
    assert "features.shell_tool=false" in args
    assert "--sandbox" in args
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert "-a" in args
    assert args[args.index("-a") + 1] == "never"
    assert "--ephemeral" in args
    assert args[-1] == "system instructions\n\n---\n\nuser question"
    assert captured["kwargs"]["stdout"] is asyncio.subprocess.PIPE
    assert captured["kwargs"]["stderr"] is asyncio.subprocess.PIPE


@pytest.mark.asyncio
async def test_installed_cli_accepts_backend_arguments_without_model_call(monkeypatch):
    executable = shutil.which("codex")
    if executable is None:
        pytest.skip("Codex CLI is not installed")
    captured = _install_fake_process(monkeypatch)
    await codex.CodexBackend().call("system", "user")

    # Exercise the real CLI parser with the generated flags, replacing the
    # prompt with --help so this can never launch a model call.
    result = subprocess.run(
        [executable, *captured["args"][1:-1], "--help"],
        env=captured["kwargs"]["env"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("use_strong", "expected_model"),
    [(False, "fast-model"), (True, "strong-model")],
)
async def test_call_routes_fast_and_strong_models(
    monkeypatch, use_strong, expected_model
):
    captured = _install_fake_process(monkeypatch)
    backend = codex.CodexBackend(
        model="legacy",
        model_fast="fast-model",
        model_strong="strong-model",
    )

    await backend.call("system", "user", use_strong=use_strong)

    assert f"model={expected_model}" in captured["args"]


@pytest.mark.asyncio
async def test_call_strips_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "metered-secret")
    monkeypatch.setenv("KEEP_ME", "present")
    captured = _install_fake_process(monkeypatch)

    await codex.CodexBackend().call("system", "user")

    env = captured["kwargs"]["env"]
    assert "OPENAI_API_KEY" not in env
    assert env["KEEP_ME"] == "present"


@pytest.mark.asyncio
async def test_call_raises_for_missing_cli(monkeypatch):
    monkeypatch.setattr(codex.shutil, "which", lambda name: None)

    with pytest.raises(LLMBackendError, match="npm i -g @openai/codex"):
        await codex.CodexBackend().call("system", "user")


@pytest.mark.asyncio
async def test_call_raises_for_nonzero_exit(monkeypatch):
    process = _FakeProcess(
        stdout=b"", stderr=b"authentication required", returncode=17
    )
    _install_fake_process(monkeypatch, process)

    with pytest.raises(
        LLMBackendError, match=r"codex exec failed \(exit 17\).*authentication"
    ):
        await codex.CodexBackend().call("system", "user")


@pytest.mark.asyncio
async def test_call_wraps_subprocess_start_error(monkeypatch):
    async def fail_to_start(*args, **kwargs):
        raise FileNotFoundError("binary disappeared")

    monkeypatch.setattr(codex.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(
        codex.asyncio, "create_subprocess_exec", fail_to_start
    )

    with pytest.raises(LLMBackendError, match="failed to start codex CLI"):
        await codex.CodexBackend().call("system", "user")


@pytest.mark.asyncio
async def test_cancelled_call_terminates_cli_process(monkeypatch):
    class _CancellableProcess:
        def __init__(self):
            self.returncode = None
            self.started = asyncio.Event()
            self.exited = asyncio.Event()
            self.terminated = False
            self.killed = False

        async def communicate(self):
            self.started.set()
            await asyncio.Event().wait()

        def terminate(self):
            self.terminated = True
            self.returncode = -15
            self.exited.set()

        def kill(self):
            self.killed = True
            self.returncode = -9
            self.exited.set()

        async def wait(self):
            await self.exited.wait()
            return self.returncode

    process = _CancellableProcess()
    _install_fake_process(monkeypatch, process)
    task = asyncio.create_task(codex.CodexBackend().call("system", "user"))
    await process.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.terminated is True
    assert process.killed is False
