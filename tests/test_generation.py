import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agents.generation import GenerationAgent, _parse_hypothesis_output
from agents.base import AgentOutputError, BaseAgent
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


def test_generation_parser_accepts_json_from_non_claude_models(config):
    h = _parse_hypothesis_output(
        '{"hypothesis": "Mechanism X", "summary": "Test X", "category": "biology"}',
        config,
        "json-model",
    )
    assert h.text == "Mechanism X"
    assert h.summary == "Test X"


@pytest.mark.parametrize("text", ["", "I cannot answer.", '{"summary": "missing"}'])
def test_generation_parser_fails_closed_on_invalid_output(config, text):
    with pytest.raises(AgentOutputError):
        _parse_hypothesis_output(text, config, "portable")


def _cfg():
    return ResearchPlanConfig(
        run_id="run1", goal="g", preferences="p",
        attributes=["Novelty"], constraints="c", safety_approved=True,
    )


def test_parse_hypothesis_accepts_markdown_decorated_labels():
    """Live Sonnet output bolds field labels (**Hypothesis:**) even when the prompt
    shows exact plain labels; 5/5 generation tasks failed on this in the first live
    run (2026-08-04). Decoration tolerance must not weaken the missing-field check."""
    from agents.generation import _parse_hypothesis_output
    text = (
        "HYPOTHESIS\n"
        "**Hypothesis:** Sildenafil-induced cGMP elevation in cardiac fibroblasts "
        "suppresses TGF-beta driven fibrotic remodelling.\n"
        "**Summary:** PDE5 inhibition as an antifibrotic strategy.\n"
        "**Category:** Cardiac fibrosis\n"
    )
    h = _parse_hypothesis_output(text, _cfg(), "debate")
    assert h.summary == "PDE5 inhibition as an antifibrotic strategy."
    assert h.category == "Cardiac fibrosis"
    assert "TGF-beta" in h.text


def test_parse_hypothesis_still_rejects_missing_fields():
    from agents.generation import _parse_hypothesis_output
    import pytest as _pytest
    from agents.base import AgentOutputError
    with _pytest.raises(AgentOutputError):
        _parse_hypothesis_output("**Hypothesis:** only, no summary", _cfg(), "debate")
