import json

import httpx
import pytest

from tools import openai_compatible
from tools.llm import LLMBackendError


def _install_transport(monkeypatch, handler):
    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    clients = []

    def make_client(**kwargs):
        client = real_async_client(transport=transport, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(openai_compatible.httpx, "AsyncClient", make_client)
    return clients


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("use_strong", "expected_model"),
    [(False, "fast-model"), (True, "strong-model")],
)
async def test_call_sends_chat_completion_request(
    monkeypatch, use_strong, expected_model
):
    monkeypatch.setenv("COMPAT_API_KEY", "test-secret")
    captured = {}

    def handler(request):
        captured["request"] = request
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "model answer"}}]},
        )

    _install_transport(monkeypatch, handler)
    backend = openai_compatible.OpenAICompatibleBackend(
        base_url="https://models.example/v1/",
        api_key_env="COMPAT_API_KEY",
        model_strong="strong-model",
        model_fast="fast-model",
        timeout_seconds=9,
    )

    result = await backend.call(
        "system instructions",
        "user question",
        use_strong=use_strong,
        max_tokens=321,
    )

    assert result == "model answer"
    request = captured["request"]
    assert str(request.url) == "https://models.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-secret"
    body = json.loads(request.content)
    assert body == {
        "model": expected_model,
        "messages": [
            {"role": "system", "content": "system instructions"},
            {"role": "user", "content": "user question"},
        ],
        "max_tokens": 321,
    }
    await backend.aclose()


@pytest.mark.asyncio
async def test_call_supports_unauthenticated_local_server(monkeypatch):
    captured = {}

    def handler(request):
        captured["request"] = request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": [{"type": "text", "text": "local"}]}}
                ]
            },
        )

    _install_transport(monkeypatch, handler)
    backend = openai_compatible.OpenAICompatibleBackend(
        base_url="http://localhost:8000/v1",
        api_key_env=None,
        model_strong="local-large",
        model_fast="local-small",
    )

    assert await backend.call("", "hello") == "local"
    assert "authorization" not in captured["request"].headers
    await backend.aclose()


@pytest.mark.asyncio
async def test_call_reuses_connection_pool_across_agents(monkeypatch):
    monkeypatch.setenv("COMPAT_API_KEY", "test-secret")

    def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "answer"}}]}
        )

    clients = _install_transport(monkeypatch, handler)
    backend = openai_compatible.OpenAICompatibleBackend(
        base_url="https://models.example/v1",
        api_key_env="COMPAT_API_KEY",
        model_strong="large",
        model_fast="small",
    )

    await backend.call("system one", "user one")
    await backend.call("system two", "user two")

    assert len(clients) == 1
    await backend.aclose()


@pytest.mark.asyncio
async def test_call_requires_configured_api_key(monkeypatch):
    monkeypatch.delenv("MISSING_COMPAT_KEY", raising=False)
    backend = openai_compatible.OpenAICompatibleBackend(
        base_url="https://models.example/v1",
        api_key_env="MISSING_COMPAT_KEY",
        model_strong="large",
        model_fast="small",
    )

    with pytest.raises(LLMBackendError, match="MISSING_COMPAT_KEY is not set"):
        await backend.call("system", "user")


@pytest.mark.asyncio
async def test_call_wraps_http_error(monkeypatch):
    monkeypatch.setenv("COMPAT_API_KEY", "test-secret")

    def handler(request):
        return httpx.Response(429, text="rate limit reached")

    _install_transport(monkeypatch, handler)
    backend = openai_compatible.OpenAICompatibleBackend(
        base_url="https://models.example/v1",
        api_key_env="COMPAT_API_KEY",
        model_strong="large",
        model_fast="small",
    )

    with pytest.raises(
        LLMBackendError, match=r"HTTP 429.*rate limit reached"
    ):
        await backend.call("system", "user")
    await backend.aclose()


@pytest.mark.asyncio
async def test_call_rejects_malformed_response(monkeypatch):
    monkeypatch.setenv("COMPAT_API_KEY", "test-secret")

    def handler(request):
        return httpx.Response(200, json={"unexpected": True})

    _install_transport(monkeypatch, handler)
    backend = openai_compatible.OpenAICompatibleBackend(
        base_url="https://models.example/v1",
        api_key_env="COMPAT_API_KEY",
        model_strong="large",
        model_fast="small",
    )

    with pytest.raises(LLMBackendError, match=r"choices\[0\]"):
        await backend.call("system", "user")
    await backend.aclose()
