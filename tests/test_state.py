import pytest
import aiosqlite
from core.state import StateStore
from core.models import Hypothesis, ResearchPlanConfig, Review, TournamentMatch


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = StateStore(db_path)
    await store.init_db()
    return store


async def test_save_and_get_hypothesis(store):
    h = Hypothesis(
        id="h1", run_id="run1", text="full text", summary="short",
        generation_method="literature", source="system",
    )
    await store.save_hypothesis(h)
    result = await store.get_hypothesis("h1")
    assert result is not None
    assert result.id == "h1"
    assert result.elo_rating == 1200.0
    assert result.annotations == []


async def test_update_elo(store):
    h = Hypothesis(
        id="h2", run_id="run1", text="text", summary="s",
        generation_method="debate", source="system",
    )
    await store.save_hypothesis(h)
    await store.update_elo("h2", 1350.5)
    result = await store.get_hypothesis("h2")
    assert result.elo_rating == 1350.5


async def test_list_active_hypotheses(store):
    for i in range(3):
        h = Hypothesis(
            id=f"h{i}", run_id="run1", text="t", summary="s",
            generation_method="literature", source="system",
        )
        await store.save_hypothesis(h)
    # Reject one
    await store.set_hypothesis_status("h0", "rejected")
    active = await store.list_hypotheses("run1", status="active")
    assert len(active) == 2


async def test_save_and_list_reviews(store):
    h = Hypothesis(
        id="h1", run_id="run1", text="t", summary="s",
        generation_method="literature", source="system",
    )
    await store.save_hypothesis(h)
    r = Review(id="r1", hypothesis_id="h1", tier=1, critique="looks ok", verdict="passed")
    await store.save_review(r)
    reviews = await store.list_reviews("h1")
    assert len(reviews) == 1
    assert reviews[0].verdict == "passed"


async def test_save_and_list_matches(store):
    match = TournamentMatch(
        id="m1", run_id="run1", h1_id="h1", h2_id="h2",
        winner_id="h1", match_type="single_turn",
        elo_before_h1=1200.0, elo_before_h2=1200.0,
        elo_after_h1=1216.0, elo_after_h2=1184.0,
    )
    await store.save_match(match)
    matches = await store.list_matches("run1")
    assert len(matches) == 1
    assert matches[0].winner_id == "h1"


async def test_save_match_and_elos_atomic(store):
    from core.models import Hypothesis, TournamentMatch
    h1 = Hypothesis(id="ha", run_id="run1", text="t", summary="s",
                    generation_method="debate", source="system")
    h2 = Hypothesis(id="hb", run_id="run1", text="t", summary="s",
                    generation_method="debate", source="system")
    await store.save_hypothesis(h1)
    await store.save_hypothesis(h2)
    match = TournamentMatch(
        id="m1", run_id="run1", h1_id="ha", h2_id="hb",
        winner_id="ha", match_type="single_turn",
        elo_before_h1=1200.0, elo_before_h2=1200.0,
        elo_after_h1=1216.0, elo_after_h2=1184.0,
    )
    await store.save_match_and_elos(match)
    matches = await store.list_matches("run1")
    assert len(matches) == 1
    ha = await store.get_hypothesis("ha")
    hb = await store.get_hypothesis("hb")
    assert ha.elo_rating == 1216.0
    assert hb.elo_rating == 1184.0
