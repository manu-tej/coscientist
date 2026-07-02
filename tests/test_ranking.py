import logging
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agents.ranking import RankingAgent, _parse_winner
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


@pytest.mark.asyncio
async def test_elo_persisted_when_store_provided(config, mock_base, tmp_path):
    from core.state import StateStore

    store = StateStore(str(tmp_path / "test.db"))
    await store.init_db()

    h1 = Hypothesis(
        id="h1", run_id="run1", text="text h1", summary="s1",
        generation_method="debate", source="system", elo_rating=1200.0,
    )
    h2 = Hypothesis(
        id="h2", run_id="run1", text="text h2", summary="s2",
        generation_method="debate", source="system", elo_rating=1200.0,
    )
    await store.save_hypothesis(h1)
    await store.save_hypothesis(h2)

    agent_with_store = RankingAgent(base=mock_base, store=store)
    await agent_with_store.run_single_turn_match(h1, h2, config, "r1", "r2")

    updated_h1 = await store.get_hypothesis("h1")
    assert updated_h1.elo_rating != 1200.0


def test_parse_winner_logs_warning_on_failure(caplog):
    with caplog.at_level(logging.WARNING, logger="agents.ranking"):
        result = _parse_winner("completely unrecognized output", "h1", "h2")
    assert result is None
    assert "Could not parse winner" in caplog.text


def test_parse_winner_returns_none_on_unparseable_text():
    assert _parse_winner("these are both interesting", "h1", "h2") is None


async def test_unparseable_single_turn_match_is_void(config, mock_base):
    mock_base.call_claude = AsyncMock(return_value="these are both interesting")
    agent = RankingAgent(base=mock_base)
    h1 = make_h("h1", 1200.0)
    h2 = make_h("h2", 1200.0)
    match = await agent.run_single_turn_match(h1, h2, config, review_1="r1", review_2="r2")
    assert match.winner_id is None
    assert h1.elo_rating == 1200.0
    assert h2.elo_rating == 1200.0
    assert match.elo_after_h1 == 1200.0
    assert match.elo_after_h2 == 1200.0


async def test_unparseable_multi_turn_match_is_void(config, mock_base):
    mock_base.run_turn_loop = AsyncMock(return_value=("no clear verdict here", ["turn1"]))
    agent = RankingAgent(base=mock_base)
    h1 = make_h("h1", 1400.0)
    h2 = make_h("h2", 1400.0)
    match = await agent.run_multi_turn_match(h1, h2, config, review_1="r1", review_2="r2")
    assert match.winner_id is None
    assert h1.elo_rating == 1400.0
    assert h2.elo_rating == 1400.0


async def test_unparseable_match_not_persisted_to_store(config, mock_base, tmp_path):
    from core.state import StateStore

    store = StateStore(str(tmp_path / "test.db"))
    await store.init_db()

    mock_base.call_claude = AsyncMock(return_value="these are both interesting")
    h1 = make_h("h1", 1200.0)
    h2 = make_h("h2", 1200.0)
    await store.save_hypothesis(h1)
    await store.save_hypothesis(h2)

    agent = RankingAgent(base=mock_base, store=store)
    await agent.run_single_turn_match(h1, h2, config, "r1", "r2")

    assert (await store.get_hypothesis("h1")).elo_rating == 1200.0
    assert (await store.get_hypothesis("h2")).elo_rating == 1200.0
