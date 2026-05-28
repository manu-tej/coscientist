import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agents.ranking import RankingAgent
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig


@pytest.fixture
def config():
    return ResearchPlanConfig(
        run_id="run1", goal="ALS", preferences="Novel",
        attributes=["Novelty"], constraints="testable", safety_approved=True,
    )


def make_h(id_, elo):
    return Hypothesis(
        id=id_, run_id="run1", text=f"text {id_}", summary=f"summary {id_}",
        generation_method="debate", source="system", elo_rating=elo,
    )


@pytest.fixture
def mock_base():
    client = MagicMock()
    base = BaseAgent(client=client, prompts_dir=Path(__file__).parent.parent / "prompts")
    base.call_claude = AsyncMock(return_value="better hypothesis: 1")
    base.run_turn_loop = AsyncMock(return_value=(
        "The debate concludes. better idea: 1",
        ["turn1", "better idea: 1"],
    ))
    return base


@pytest.fixture
def agent(mock_base):
    return RankingAgent(base=mock_base, elo_k=32.0, multi_turn_threshold=1350.0)


async def test_single_turn_match_h1_wins(agent, config):
    h1 = make_h("h1", 1200.0)
    h2 = make_h("h2", 1200.0)
    match = await agent.run_single_turn_match(h1, h2, config, review_1="ok", review_2="ok")
    assert match.winner_id == "h1"
    assert match.elo_after_h1 > 1200.0
    assert match.elo_after_h2 < 1200.0
    assert match.match_type == "single_turn"


async def test_multi_turn_match_h1_wins(agent, config):
    h1 = make_h("h1", 1400.0)
    h2 = make_h("h2", 1400.0)
    match = await agent.run_multi_turn_match(h1, h2, config, review_1="ok", review_2="ok")
    assert match.winner_id == "h1"
    assert match.match_type == "multi_turn"
    assert match.debate_transcript is not None


async def test_selects_multi_turn_for_high_elo(agent, config):
    h1 = make_h("h1", 1400.0)
    h2 = make_h("h2", 1400.0)
    match = await agent.run_match(h1, h2, config, review_1="r1", review_2="r2")
    assert match.match_type == "multi_turn"


async def test_selects_single_turn_for_low_elo(agent, config):
    h1 = make_h("h1", 1200.0)
    h2 = make_h("h2", 1200.0)
    match = await agent.run_match(h1, h2, config, review_1="r1", review_2="r2")
    assert match.match_type == "single_turn"
