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


async def test_save_match_and_elos_applies_delta_not_absolute(store):
    """BUG 7: elo_after_* was computed from a snapshot at ranking-task start.
    If a concurrent match already moved a rating, the write must apply the
    DELTA (after - before) so both matches' changes compose — not overwrite
    with the stale absolute elo_after (lost update)."""
    from core.models import Hypothesis, TournamentMatch
    h1 = Hypothesis(id="ha", run_id="run1", text="t", summary="s",
                    generation_method="debate", source="system")
    h2 = Hypothesis(id="hb", run_id="run1", text="t", summary="s",
                    generation_method="debate", source="system")
    await store.save_hypothesis(h1)
    await store.save_hypothesis(h2)
    # Simulate a concurrent match that moved the ratings AFTER this match
    # snapshotted elo_before = 1200.
    await store.update_elo("ha", 1250.0)
    await store.update_elo("hb", 1180.0)
    match = TournamentMatch(
        id="m1", run_id="run1", h1_id="ha", h2_id="hb",
        winner_id="ha", match_type="single_turn",
        elo_before_h1=1200.0, elo_before_h2=1200.0,
        elo_after_h1=1216.0, elo_after_h2=1184.0,
    )
    await store.save_match_and_elos(match)
    ha = await store.get_hypothesis("ha")
    hb = await store.get_hypothesis("hb")
    # current + (after - before), NOT the stale absolute elo_after
    assert ha.elo_rating == pytest.approx(1250.0 + 16.0)
    assert hb.elo_rating == pytest.approx(1180.0 - 16.0)
    assert ha.elo_rating != 1216.0
    assert hb.elo_rating != 1184.0


async def test_save_match_and_elos_void_match_leaves_ratings_unchanged(store):
    """A void match (winner None, elo_after == elo_before) has delta 0 and
    must not move ratings, even ones that changed since the snapshot."""
    from core.models import Hypothesis, TournamentMatch
    h1 = Hypothesis(id="ha", run_id="run1", text="t", summary="s",
                    generation_method="debate", source="system")
    h2 = Hypothesis(id="hb", run_id="run1", text="t", summary="s",
                    generation_method="debate", source="system")
    await store.save_hypothesis(h1)
    await store.save_hypothesis(h2)
    await store.update_elo("ha", 1250.0)
    match = TournamentMatch(
        id="m1", run_id="run1", h1_id="ha", h2_id="hb",
        winner_id=None, match_type="single_turn",
        elo_before_h1=1200.0, elo_before_h2=1200.0,
        elo_after_h1=1200.0, elo_after_h2=1200.0,
    )
    await store.save_match_and_elos(match)
    assert (await store.get_hypothesis("ha")).elo_rating == 1250.0
    assert (await store.get_hypothesis("hb")).elo_rating == 1200.0
    assert len(await store.list_matches("run1")) == 1


async def test_try_claim_review_first_wins(store):
    from core.models import Hypothesis
    h = Hypothesis(id="hc", run_id="run1", text="t", summary="s",
                   generation_method="debate", source="system")
    await store.save_hypothesis(h)
    first = await store.try_claim_review("hc")
    second = await store.try_claim_review("hc")
    assert first is True
    assert second is False


async def test_try_claim_review_distinct_hypotheses(store):
    a = await store.try_claim_review("ha")
    b = await store.try_claim_review("hb")
    assert a is True
    assert b is True


async def test_release_claim_allows_reclaim(store):
    """BUG 10b: claims must be releasable, or a crashed reflection worker
    holds the claim forever and the hypothesis starves."""
    assert await store.try_claim_review("hc") is True
    assert await store.try_claim_review("hc") is False
    await store.release_claim("hc")
    assert await store.try_claim_review("hc") is True


async def test_release_claim_on_unclaimed_hypothesis_is_noop(store):
    await store.release_claim("never-claimed")
    assert await store.try_claim_review("never-claimed") is True


async def test_init_db_enables_wal(store):
    """BUG 11: init_db must switch the DB file to WAL journal mode so
    concurrent readers/writers don't hit 'database is locked'."""
    async with aiosqlite.connect(store.db_path) as db:
        async with db.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
    assert row[0].lower() == "wal"


async def test_connect_helper_sets_busy_timeout(store):
    """BUG 11: connections opened via the store wait (busy_timeout) for a
    lock instead of failing immediately."""
    async with store._connect() as db:
        async with db.execute("PRAGMA busy_timeout") as cursor:
            row = await cursor.fetchone()
    assert row[0] == 5000
