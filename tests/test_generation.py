import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agents.generation import GenerationAgent
from agents.base import BaseAgent
from core.models import ResearchPlanConfig, GenerationStrategy


@pytest.fixture
def config():
    return ResearchPlanConfig(
        run_id="run1",
        goal="Develop a novel hypothesis for ALS mechanisms",
        preferences="Focus on novel, testable hypotheses",
        attributes=["Novelty", "Feasibility"],
        constraints="Must be testable in vitro",
        safety_approved=True,
    )


@pytest.fixture
def mock_base():
    client = MagicMock()
    base = BaseAgent(client=client, prompts_dir=Path(__file__).parent.parent / "prompts")
    base.call_claude = AsyncMock(return_value=(
        "Introduction: ALS context\n"
        "Recent Findings: TDP-43\n"
        "Hypothesis: Stress induces PTMs on Nup98\n"
        "Rationale and Specificity: Novel because...\n"
        "Experimental Design: Use iPSC cells\n"
        "Summary: Stress-induced PTMs on nucleoporins cause ALS\n"
        "Category: Neurodegeneration"
    ))
    base.run_turn_loop = AsyncMock(return_value=(
        "HYPOTHESIS\nHypothesis: Stress induces PTMs on Nup98\nSummary: PTM hypothesis\nCategory: Neurodegeneration",
        ["turn1", "HYPOTHESIS\nHypothesis: Stress induces PTMs\nSummary: PTM hypothesis\nCategory: Neurodegeneration"]
    ))
    return base


@pytest.fixture
def agent(mock_base):
    return GenerationAgent(base=mock_base)


async def test_literature_strategy_returns_hypothesis(agent, config):
    articles = "Article 1: TDP-43 study..."
    h = await agent.run_literature(config, articles_with_reasoning=articles)
    assert h.text != ""
    assert h.summary != ""
    assert h.generation_method == "literature"
    assert h.source == "system"
    assert h.run_id == "run1"


async def test_debate_strategy_returns_hypothesis(agent, config):
    h = await agent.run_debate(config, reviews_overview="No prior meta-review")
    assert h.generation_method == "debate"
    # Verify run_turn_loop was called with the right termination signal
    call_args = agent.base.run_turn_loop.call_args
    assert call_args.kwargs.get("termination_signal") == "HYPOTHESIS" or \
           "HYPOTHESIS" in str(call_args)


async def test_hypothesis_has_uuid(agent, config):
    h1 = await agent.run_literature(config, articles_with_reasoning="articles")
    h2 = await agent.run_literature(config, articles_with_reasoning="articles")
    assert h1.id != h2.id


async def test_assumptions_strategy_returns_hypothesis(agent, config):
    h = await agent.run_assumptions(config)
    assert h.generation_method == "assumptions"
    assert h.run_id == "run1"
