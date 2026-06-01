# tests/test_config_parser.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from core.config_parser import ConfigParser
from core.models import ResearchPlanConfig
from tools.claude import ClaudeClient

PROMPTS = Path(__file__).parent.parent / "prompts"


@pytest.fixture
def client():
    return MagicMock(spec=ClaudeClient)


@pytest.fixture
def parser(client):
    return ConfigParser(client=client, prompts_dir=PROMPTS)


async def test_parse_extracts_fields(parser, client):
    client.call = AsyncMock(return_value=(
        "PREFERENCES: Focus on novel molecular mechanisms\n"
        "ATTRIBUTES: Novelty, Feasibility\n"
        "CONSTRAINTS: Must be testable in vitro; should be novel"
    ))
    config = await parser.parse(run_id="run1", goal="Explain ALS mechanisms")
    assert config.run_id == "run1"
    assert config.goal == "Explain ALS mechanisms"
    assert config.preferences == "Focus on novel molecular mechanisms"
    assert config.attributes == ["Novelty", "Feasibility"]
    assert "testable in vitro" in config.constraints
    assert config.idea_attributes == "Novelty and Feasibility"


async def test_safety_review_approved(parser, client):
    client.call = AsyncMock(return_value="DECISION: APPROVED\nREASON: Legitimate biomedical research.")
    approved, reason = await parser.safety_review("Find drug repurposing candidates for AML")
    assert approved is True
    assert "Legitimate" in reason


async def test_safety_review_rejected(parser, client):
    client.call = AsyncMock(return_value="DECISION: REJECTED\nREASON: Primarily aims to create a harmful agent.")
    approved, reason = await parser.safety_review("Design a more lethal pathogen")
    assert approved is False
    assert "harmful" in reason.lower()


async def test_parse_and_review_combines(parser, client):
    # safety first (approved), then parse
    client.call = AsyncMock(side_effect=[
        "DECISION: APPROVED\nREASON: Legitimate.",
        "PREFERENCES: Focus on novelty\nATTRIBUTES: Novelty\nCONSTRAINTS: Testable",
    ])
    config = await parser.parse_and_review(run_id="run2", goal="Study X")
    assert config is not None
    assert config.safety_approved is True
    assert config.attributes == ["Novelty"]


async def test_parse_and_review_rejects_unsafe(parser, client):
    client.call = AsyncMock(return_value="DECISION: REJECTED\nREASON: Unsafe.")
    config = await parser.parse_and_review(run_id="run3", goal="Make a weapon")
    assert config is None
