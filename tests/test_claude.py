import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from tools.claude import ClaudeClient


@pytest.fixture
def client():
    return ClaudeClient(model_strong="claude-opus-4-7", model_fast="claude-sonnet-4-6")


async def test_call_returns_text(client):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Test response")]

    with patch.object(client._client.messages, "create", new=AsyncMock(return_value=mock_response)):
        result = await client.call("You are helpful.", "What is ALS?", use_strong=False)

    assert result == "Test response"


async def test_strong_model_used_for_deep_tasks(client):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Deep analysis")]

    with patch.object(client._client.messages, "create", new=AsyncMock(return_value=mock_response)) as mock_create:
        await client.call("system", "prompt", use_strong=True)

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "claude-opus-4-7"


async def test_fast_model_used_by_default(client):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Fast response")]

    with patch.object(client._client.messages, "create", new=AsyncMock(return_value=mock_response)) as mock_create:
        await client.call("system", "prompt", use_strong=False)

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"


async def test_cache_control_applied_to_system_prompt(client):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="cached")]

    with patch.object(client._client.messages, "create", new=AsyncMock(return_value=mock_response)) as mock_create:
        await client.call("Be concise.", "Explain ALS.")

    system_arg = mock_create.call_args.kwargs["system"]
    assert isinstance(system_arg, list)
    assert system_arg[0]["cache_control"] == {"type": "ephemeral"}
    assert system_arg[0]["text"] == "Be concise."
