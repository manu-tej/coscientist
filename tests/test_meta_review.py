import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agents.meta_review import MetaReviewAgent
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig


@pytest.fixture
def config():
    return ResearchPlanConfig(
        run_id="run1", goal="ALS mechanisms", preferences="Novel",
        attributes=["Novelty"], constraints="testable", safety_approved=True,
    )


@pytest.fixture
def hypotheses():
    return [
        Hypothesis(id=f"h{i}", run_id="run1", text=f"text {i}", summary=f"summary {i}",
                   generation_method="debate", source="system", elo_rating=1200.0 + i * 50)
        for i in range(3)
    ]


@pytest.fixture
def mock_base():
    client = MagicMock()
    base = BaseAgent(client=client, prompts_dir=Path(__file__).parent.parent / "prompts")
    base.call_claude = AsyncMock(return_value=(
        "I. Core Hypothesis: recurring issue is lack of specificity.\n"
        "II. Experimental: model limitations noted."
    ))
    return base


@pytest.fixture
def agent(mock_base):
    return MetaReviewAgent(base=mock_base)


async def test_run_meta_critique(agent, config):
    reviews_text = "Review 1: lacks specificity\nReview 2: needs better controls"
    critique = await agent.run_meta_critique(config, reviews_text)
    assert len(critique) > 0
    assert isinstance(critique, str)


async def test_run_research_overview(agent, config, hypotheses, mock_base):
    mock_base.call_claude = AsyncMock(return_value=(
        "[Main Research Directions]\nDirection: PTM mechanisms\nRationale: Novel"
    ))
    overview = await agent.run_research_overview(config, hypotheses)
    assert len(overview) > 0
    assert isinstance(overview, str)


async def test_research_overview_uses_top_hypotheses_sorted_by_elo(agent, config, hypotheses, mock_base):
    mock_base.call_claude = AsyncMock(return_value="overview text")
    await agent.run_research_overview(config, hypotheses)
    call_args = mock_base.call_claude.call_args
    prompt_text = call_args[0][1]  # user prompt
    # Highest Elo hypothesis (h2, elo=1300) should appear first in prompt
    assert "summary 2" in prompt_text


async def test_run_research_contacts(agent, config, mock_base):
    mock_base.call_claude = AsyncMock(return_value=(
        "Research Direction: ALS\n[Smith Lab]: Studies TDP-43"
    ))
    cited = "TDP-43 study: http://example.com"
    contacts = await agent.run_research_contacts(config, cited)
    assert len(contacts) > 0
