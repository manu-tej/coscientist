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


async def test_ranking_agent_never_calls_update_elo(config, mock_base):
    """BUG 8: the agent computes new ratings but must NOT persist them —
    persistence happens exactly once, in StateStore.save_match_and_elos."""
    store = MagicMock()
    store.update_elo = AsyncMock()
    agent = RankingAgent(base=mock_base, store=store)

    h1 = make_h("h1", 1200.0)
    h2 = make_h("h2", 1200.0)
    match = await agent.run_single_turn_match(h1, h2, config, "r1", "r2")
    store.update_elo.assert_not_called()
    # In-memory computation still happens and is recorded on the match.
    assert match.elo_after_h1 > match.elo_before_h1
    assert h1.elo_rating == match.elo_after_h1

    h3 = make_h("h3", 1400.0)
    h4 = make_h("h4", 1400.0)
    await agent.run_multi_turn_match(h3, h4, config, "r1", "r2")
    store.update_elo.assert_not_called()


@pytest.mark.asyncio
async def test_elo_persisted_exactly_once_via_save_match_and_elos(config, mock_base, tmp_path):
    """BUG 8: after run_match the store is untouched; save_match_and_elos then
    applies the Elo change exactly once (not the double write of agent
    update_elo + save_match_and_elos)."""
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
    match = await agent_with_store.run_single_turn_match(h1, h2, config, "r1", "r2")

    # Agent must not have persisted anything yet.
    assert (await store.get_hypothesis("h1")).elo_rating == 1200.0
    assert (await store.get_hypothesis("h2")).elo_rating == 1200.0

    await store.save_match_and_elos(match)

    # Exactly one application of the delta (1200 + delta == elo_after).
    assert (await store.get_hypothesis("h1")).elo_rating == pytest.approx(match.elo_after_h1)
    assert (await store.get_hypothesis("h2")).elo_rating == pytest.approx(match.elo_after_h2)


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
