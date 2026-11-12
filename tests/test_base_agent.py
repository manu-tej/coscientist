import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agents.base import BaseAgent
from tools.claude import ClaudeClient


@pytest.fixture
def client(mocker):
    c = mocker.MagicMock(spec=ClaudeClient)
    c.call = AsyncMock(return_value="test response")
    return c


@pytest.fixture
def agent(tmp_path, client):
    # Create a test prompt file
    prompt_dir = tmp_path / "prompts" / "test"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "hello.txt").write_text("Hello {name}, goal is {goal}.")
    # Create a debate-style prompt with {transcript}
    debate_dir = tmp_path / "prompts" / "debate"
    debate_dir.mkdir(parents=True)
    (debate_dir / "test.txt").write_text("Goal: {goal}\n#BEGIN TRANSCRIPT#\n{transcript}\n#END TRANSCRIPT#\nYour Turn:")
    return BaseAgent(client=client, prompts_dir=tmp_path / "prompts")


def test_render_prompt(agent):
    result = agent.render_prompt("test/hello", name="Alice", goal="cure ALS")
    assert result == "Hello Alice, goal is cure ALS."


def test_render_prompt_preserves_json_braces_in_values(agent):
    result = agent.render_prompt(
        "test/hello",
        name="Alice",
        goal='inspect {"drug": "sildenafil", "score": 0.9}',
    )
    assert result == (
        'Hello Alice, goal is inspect {"drug": "sildenafil", "score": 0.9}.'
    )


async def test_call_claude_delegates(agent, client):
    result = await agent.call_claude("sys", "user")
    client.call.assert_called_once_with("sys", "user", use_strong=False, max_tokens=8192)
    assert result == "test response"


async def test_turn_loop_terminates_on_signal(agent, client):
    # First turn: no signal. Second turn: contains termination signal.
    client.call = AsyncMock(side_effect=[
        "I propose three ideas...",
        "After debate... HYPOTHESIS\nFinal hypothesis text here.",
    ])
    final, transcript = await agent.run_turn_loop(
        template_name="debate/test",
        variables={"goal": "test"},
        transcript_key="transcript",
        termination_signal="HYPOTHESIS",
        max_turns=10,
    )
    assert "HYPOTHESIS" in final
    assert len(transcript) == 2


async def test_turn_loop_respects_max_turns(agent, client):
    client.call = AsyncMock(return_value="No termination signal here.")
    final, transcript = await agent.run_turn_loop(
        template_name="debate/test",
        variables={"goal": "test"},
        transcript_key="transcript",
        termination_signal="HYPOTHESIS",
        max_turns=3,
    )
    assert len(transcript) == 3


async def test_turn_loop_ignores_incidental_mention_of_the_signal():
    """A model discussing the signal mid-debate must not end the loop.

    The debate prompt itself contains the word HYPOTHESIS, so models echo it while
    still deliberating; ending there yields an unlabelled turn that the parser
    discards, losing the whole discussion. Callers guard with is_complete.
    """
    from unittest.mock import AsyncMock, MagicMock
    from agents.base import BaseAgent
    from pathlib import Path

    base = BaseAgent(client=MagicMock(), prompts_dir=Path("prompts"))
    base.render_prompt = MagicMock(return_value="p")
    base.call_claude = AsyncMock(side_effect=[
        "Round 1: we will converge on a HYPOTHESIS after more debate.",
        "HYPOTHESIS\nHypothesis: the real one\nSummary: done\n",
    ])
    last, transcript = await base.run_turn_loop(
        template_name="generation/debate", variables={}, transcript_key="transcript",
        termination_signal="HYPOTHESIS", max_turns=5,
        is_complete=lambda r: "Hypothesis:" in r and "Summary:" in r,
    )
    assert len(transcript) == 2, "loop ended on the incidental mention"
    assert "Hypothesis: the real one" in last


async def test_turn_loop_terminates_when_contract_satisfied():
    from unittest.mock import AsyncMock, MagicMock
    from agents.base import BaseAgent
    from pathlib import Path

    base = BaseAgent(client=MagicMock(), prompts_dir=Path("prompts"))
    base.render_prompt = MagicMock(return_value="p")
    base.call_claude = AsyncMock(side_effect=[
        "HYPOTHESIS\nHypothesis: x\nSummary: y\n",
        "should never be requested",
    ])
    _, transcript = await base.run_turn_loop(
        template_name="generation/debate", variables={}, transcript_key="transcript",
        termination_signal="HYPOTHESIS", max_turns=5,
        is_complete=lambda r: "Hypothesis:" in r and "Summary:" in r,
    )
    assert len(transcript) == 1
