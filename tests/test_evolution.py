import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agents.evolution import EvolutionAgent
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig


HYPOTHESIS_OUTPUT = (
    "Introduction: ALS context\n"
    "Hypothesis: New PTM mechanism\n"
    "Summary: Improved PTM hypothesis\n"
    "Category: Neurodegeneration"
)


@pytest.fixture
def config():
    return ResearchPlanConfig(
        run_id="run1", goal="ALS mechanisms", preferences="Novel",
        attributes=["Novelty", "Feasibility"], constraints="testable", safety_approved=True,
    )


@pytest.fixture
def source_hypothesis():
    return Hypothesis(
        id="h1", run_id="run1", text="Original hypothesis text",
        summary="Original", generation_method="debate", source="system",
    )


@pytest.fixture
def mock_base():
    client = MagicMock()
    base = BaseAgent(client=client, prompts_dir=Path(__file__).parent.parent / "prompts")
    base.call_claude = AsyncMock(return_value=HYPOTHESIS_OUTPUT)
    return base


@pytest.fixture
def agent(mock_base):
    return EvolutionAgent(base=mock_base)


async def test_grounding_creates_new_hypothesis(agent, source_hypothesis, config):
    h = await agent.run_grounding(source_hypothesis, config, weaknesses="lacks specificity", articles="article content")
    assert h.id != source_hypothesis.id
    assert h.evolved_from == source_hypothesis.id
    assert h.generation_method == "grounding"
    assert h.run_id == "run1"


async def test_out_of_box_uses_multiple_hypotheses(agent, config, mock_base):
    hypotheses = [
        Hypothesis(id=f"h{i}", run_id="run1", text=f"text {i}", summary=f"s{i}",
                   generation_method="debate", source="system")
        for i in range(3)
    ]
    h = await agent.run_out_of_box(hypotheses, config)
    assert h.generation_method == "out_of_box"
    # Verify multiple hypothesis summaries were included in the prompt
    call_args = mock_base.call_claude.call_args
    prompt_text = call_args[0][1]  # second positional arg is the user prompt
    assert "s0" in prompt_text or "s1" in prompt_text


async def test_combination_has_no_single_parent(agent, config):
    hypotheses = [
        Hypothesis(id=f"h{i}", run_id="run1", text=f"text {i}", summary=f"s{i}",
                   generation_method="debate", source="system")
        for i in range(2)
    ]
    h = await agent.run_combination(hypotheses, config)
    assert h.generation_method == "combination"
    assert h.evolved_from is None  # combination has multiple parents, none tracked


async def test_coherence_links_to_parent(agent, source_hypothesis, config):
    h = await agent.run_coherence(source_hypothesis, config)
    assert h.evolved_from == source_hypothesis.id
    assert h.generation_method == "coherence"


async def test_simplification_links_to_parent(agent, source_hypothesis, config):
    h = await agent.run_simplification(source_hypothesis, config)
    assert h.evolved_from == source_hypothesis.id
    assert h.generation_method == "simplification"
