import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from agents.reflection import ReflectionAgent, _parse_verdict, _extract_observation
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig, Review


@pytest.fixture
def config():
    return ResearchPlanConfig(
        run_id="run1",
        goal="ALS mechanisms",
        preferences="Novel and feasible",
        attributes=["Novelty", "Feasibility"],
        constraints="testable",
        safety_approved=True,
    )


@pytest.fixture
def hypothesis():
    return Hypothesis(
        id="h1", run_id="run1", text="full text of hypothesis",
        summary="A hypothesis about PTMs", generation_method="debate", source="system",
    )


@pytest.fixture
def mock_base():
    client = MagicMock()
    base = BaseAgent(client=client, prompts_dir=Path(__file__).parent.parent / "prompts")
    base.call_claude = AsyncMock(return_value=(
        "Correctness: HIGH - well-reasoned\n"
        "Quality: HIGH - specific\n"
        "Novelty: HIGH - novel\n"
        "Safety: SAFE - no issues\n"
        "Critique:\n- Strong hypothesis\n"
        "Verdict: PASSED\n"
        "Reason for verdict: Well-reasoned hypothesis."
    ))
    return base


@pytest.fixture
def agent(mock_base):
    return ReflectionAgent(base=mock_base)


async def test_initial_review_passed(agent, hypothesis, config):
    review = await agent.run_initial_review(hypothesis, config)
    assert review.tier == 1
    assert review.hypothesis_id == "h1"
    assert review.verdict == "passed"
    assert len(review.critique) > 0


async def test_initial_review_rejected_on_keyword(agent, hypothesis, config, mock_base):
    mock_base.call_claude = AsyncMock(return_value=(
        "Correctness: LOW\nQuality: LOW\nNovelty: LOW\nSafety: SAFE\n"
        "Critique:\n- Not novel\n"
        "Verdict: REJECTED\n"
        "Reason for verdict: Not novel."
    ))
    review = await agent.run_initial_review(hypothesis, config)
    assert review.verdict == "rejected"


async def test_full_review_tier_2(agent, hypothesis, config):
    review = await agent.run_full_review(hypothesis, config, articles_with_reasoning="Article 1: ...")
    assert review.tier == 2
    assert review.hypothesis_id == "h1"


async def test_observation_review_extracts_missing_piece(agent, hypothesis, config, mock_base):
    mock_base.call_claude = AsyncMock(return_value=(
        "1. Observations: widespread TDP-43 in ALS neurons\n"
        "2. Analysis: would we see this if hypothesis true? Yes\n"
        "3. Summary: would we see some observations? Yes\n"
        "4. Disproof: does some observation disprove? No\n"
        "hypothesis: missing piece"
    ))
    review, observation = await agent.run_observation_review(
        hypothesis, config, article="Article content here"
    )
    assert review.tier == 4
    assert observation is not None


async def test_simulation_review_tier_5(agent, hypothesis, config):
    review = await agent.run_simulation_review(hypothesis, config)
    assert review.tier == 5


async def test_tournament_review_tier_6(agent, hypothesis, config):
    review = await agent.run_tournament_review(
        hypothesis, config,
        prior_reviews="prior review text",
        meta_critique="meta critique",
        tournament_history="match history"
    )
    assert review.tier == 6


# --- BUG A: _parse_verdict must fail SAFE, not open ---

def test_parse_verdict_unparseable_does_not_pass():
    # Formatting drift with no clean "verdict: <token>" must NOT silently pass.
    text = "The reviewer had concerns but the format is garbled here."
    assert _parse_verdict(text) != "passed"
    assert _parse_verdict(text) == "flagged"


def test_parse_verdict_bracketed_rejected():
    assert _parse_verdict("Verdict: [REJECTED]") == "rejected"


def test_parse_verdict_markdown_rejected():
    assert _parse_verdict("**Verdict:** rejected") == "rejected"


def test_parse_verdict_plain_passed_still_passes():
    assert _parse_verdict("Verdict: PASSED\nReason: solid.") == "passed"


# --- BUG B: _extract_observation must not treat negations as observations ---

def test_extract_observation_ignores_negation():
    assert _extract_observation("This is not a missing piece.") is None


def test_extract_observation_returns_affirmative_line():
    text = "Analysis follows.\nThe missing piece is X\nDone."
    assert _extract_observation(text) == "The missing piece is X"
