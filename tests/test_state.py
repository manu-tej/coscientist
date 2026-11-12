import pytest
import asyncio
import aiosqlite
from core.state import StateStore
from core.models import (
    AbsoluteScore,
    Hypothesis,
    ResearchPlanConfig,
    Review,
    TournamentMatch,
)


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


async def test_get_grounding_refs_defaults_to_none(store):
    """The column default is NULL -- every pre-existing hypothesis (and any
    hypothesis whose generation strategy never records grounding) must read
    back as None, not an empty set, so a citation check can tell "never
    recorded" apart from "recorded as nothing was shown"."""
    h = Hypothesis(
        id="h-none", run_id="run1", text="t", summary="s",
        generation_method="debate", source="system",
    )
    await store.save_hypothesis(h)
    assert await store.get_grounding_refs("h-none") is None


async def test_set_and_get_grounding_refs_round_trips(store):
    h = Hypothesis(
        id="h-refs", run_id="run1", text="t", summary="s",
        generation_method="literature", source="system",
    )
    await store.save_hypothesis(h)
    await store.set_grounding_refs("h-refs", {"111", "222"})
    assert await store.get_grounding_refs("h-refs") == {"111", "222"}


async def test_set_grounding_refs_with_empty_set_is_distinct_from_unset(store):
    """An explicitly-recorded empty set ("we showed nothing") must not read
    back as None ("we never recorded what was shown") -- they mean opposite
    things to a citation check."""
    h = Hypothesis(
        id="h-empty", run_id="run1", text="t", summary="s",
        generation_method="literature", source="system",
    )
    await store.save_hypothesis(h)
    await store.set_grounding_refs("h-empty", set())
    refs = await store.get_grounding_refs("h-empty")
    assert refs == set()
    assert refs is not None


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


def _absolute_score(hypothesis_id: str, *, total: float = 4.0) -> AbsoluteScore:
    return AbsoluteScore(
        run_id="run1",
        hypothesis_id=hypothesis_id,
        scorer_id="absolute-test",
        axis_scores={
            "novelty": 4,
            "feasibility": 4,
            "correctness": 4,
            "impact": 4,
        },
        total=total,
    )


async def test_absolute_score_round_trip_keeps_elo_separate(store):
    hypothesis = Hypothesis(
        id="absolute-h", run_id="run1", text="candidate", summary="candidate",
        generation_method="debate", source="system",
    )
    await store.save_hypothesis(hypothesis)
    claim = await store.claim_absolute_score("absolute-h", "absolute-test")
    assert claim.state == "acquired"
    await store.publish_absolute_score(
        _absolute_score("absolute-h"), owner_token=claim.owner_token
    )

    reopened = StateStore(store.db_path)
    score = await reopened.get_absolute_score("absolute-h")
    persisted_hypothesis = await reopened.get_hypothesis("absolute-h")

    assert score == _absolute_score("absolute-h")
    assert persisted_hypothesis.elo_rating == 1200.0


async def test_absolute_score_claim_is_atomic_across_connections(store):
    await store.save_hypothesis(Hypothesis(
        id="claim-h", run_id="run1", text="candidate", summary="candidate",
        generation_method="debate", source="system",
    ))
    connections = [StateStore(store.db_path) for _ in range(8)]
    claims = await asyncio.gather(*(
        connection.claim_absolute_score("claim-h", "absolute-test")
        for connection in connections
    ))

    assert sum(claim.state == "acquired" for claim in claims) == 1
    assert sum(claim.state == "waiting" for claim in claims) == 7


async def test_absolute_score_stale_owner_cannot_publish_or_release(store):
    await store.save_hypothesis(Hypothesis(
        id="stale-h", run_id="run1", text="candidate", summary="candidate",
        generation_method="debate", source="system",
    ))
    owner = await store.claim_absolute_score(
        "stale-h", "absolute-test", lease_seconds=0.01
    )
    await asyncio.sleep(0.02)
    takeover = await store.claim_absolute_score(
        "stale-h", "absolute-test", lease_seconds=5
    )

    assert takeover.state == "acquired"
    with pytest.raises(ValueError, match="active claim"):
        await store.publish_absolute_score(
            _absolute_score("stale-h"), owner_token=owner.owner_token
        )
    assert await store.release_absolute_score_claim(
        "stale-h", "absolute-test", owner.owner_token
    ) is False

    await store.publish_absolute_score(
        _absolute_score("stale-h"), owner_token=takeover.owner_token
    )
    assert (await store.get_absolute_score("stale-h")).total == 4.0


async def test_failed_absolute_score_claim_can_be_retried(store):
    await store.save_hypothesis(Hypothesis(
        id="retry-h", run_id="run1", text="candidate", summary="candidate",
        generation_method="debate", source="system",
    ))
    first = await store.claim_absolute_score("retry-h", "absolute-test")
    assert await store.release_absolute_score_claim(
        "retry-h", "absolute-test", first.owner_token
    ) is True
    second = await store.claim_absolute_score("retry-h", "absolute-test")

    assert second.state == "acquired"
    assert second.owner_token != first.owner_token


async def test_init_db_enables_wal(store):
    """Tier-2 audit: init_db must switch the DB file to WAL journal mode so
    concurrent readers/writers don't hit 'database is locked'."""
    async with aiosqlite.connect(store.db_path) as db:
        async with db.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
    assert row[0].lower() == "wal"


async def test_connect_helper_sets_busy_timeout(store):
    """Tier-2 audit: connections opened via the store wait for a lock instead
    of failing immediately, so write contention is not a task failure."""
    async with store._connect() as db:
        async with db.execute("PRAGMA busy_timeout") as cursor:
            row = await cursor.fetchone()
    assert row[0] == 30000


async def test_claim_verdicts_round_trip(store):
    from tools.evidence import ClaimVerdict, EvidenceRecord

    await store.save_hypothesis(Hypothesis(
        id="hv", run_id="run1", text="t", summary="s",
        generation_method="debate", source="system",
    ))
    await store.save_claim_verdict("hv", ClaimVerdict(
        claim="SMAD3 binds the PDE5A promoter",
        status="supported",
        records=[EvidenceRecord(
            source="pubmed", ref="28442", title="A study", snippet="Abstract text.",
            url="https://pubmed.ncbi.nlm.nih.gov/28442/", year=2017, has_abstract=True,
        )],
        rationale="Abstract states direct occupancy.",
    ))

    stored = await store.list_claim_verdicts("hv")
    assert len(stored) == 1
    assert stored[0].status == "supported"
    assert stored[0].records[0].ref == "28442"
    assert stored[0].records[0].has_abstract is True


async def test_claim_verdicts_scoped_to_hypothesis(store):
    from tools.evidence import ClaimVerdict

    for hid in ("ha", "hb"):
        await store.save_hypothesis(Hypothesis(
            id=hid, run_id="run1", text="t", summary="s",
            generation_method="debate", source="system",
        ))
    await store.save_claim_verdict("ha", ClaimVerdict(claim="c1", status="unchecked"))
    await store.save_claim_verdict("hb", ClaimVerdict(claim="c2", status="contradicted"))

    assert [v.claim for v in await store.list_claim_verdicts("ha")] == ["c1"]
    assert [v.status for v in await store.list_claim_verdicts("hb")] == ["contradicted"]


async def test_claim_verdicts_multiple_per_hypothesis_all_persist_in_order(store):
    """A hypothesis has MANY claim verdicts: plain INSERT (not INSERT OR
    REPLACE) must let them all survive as distinct rows, with statuses
    intact. Writing several verdicts back-to-back also collides their
    created_at timestamps (SQLite's CURRENT_TIMESTAMP has 1-second
    granularity), so the returned order must come from the rowid tiebreaker
    -- insertion order -- not be arbitrary."""
    from tools.evidence import ClaimVerdict

    await store.save_hypothesis(Hypothesis(
        id="hm", run_id="run1", text="t", summary="s",
        generation_method="debate", source="system",
    ))
    await store.save_claim_verdict("hm", ClaimVerdict(claim="claim-1", status="supported"))
    await store.save_claim_verdict("hm", ClaimVerdict(claim="claim-2", status="unchecked"))
    await store.save_claim_verdict("hm", ClaimVerdict(claim="claim-3", status="supported"))

    stored = await store.list_claim_verdicts("hm")
    assert [v.claim for v in stored] == ["claim-1", "claim-2", "claim-3"]
    assert [v.status for v in stored] == ["supported", "unchecked", "supported"]


async def test_unknown_record_keys_do_not_break_reads(store):
    """Schema drift must not make old rows unreadable."""
    import json
    from tools.evidence import ClaimVerdict

    await store.save_hypothesis(Hypothesis(
        id="hz", run_id="run1", text="t", summary="s",
        generation_method="debate", source="system",
    ))
    await store.save_claim_verdict("hz", ClaimVerdict(claim="c", status="supported"))
    async with store._connect() as db:
        await db.execute(
            "UPDATE claim_verdicts SET records = ? WHERE hypothesis_id = 'hz'",
            (json.dumps([{"source": "pubmed", "ref": "1", "title": "t", "snippet": "s",
                          "url": "u", "year": 2020, "has_abstract": True,
                          "future_field": "x"}]),),
        )
        await db.commit()

    stored = await store.list_claim_verdicts("hz")
    assert stored[0].records[0].ref == "1"


async def test_init_db_adds_columns_missing_from_an_older_database(tmp_path):
    """A database created before `grounding_refs` shipped must survive an upgrade.

    CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so a new *table*
    lands on an old DB but a new *column* never does. Reproduced live on
    2026-08-09: bench.offline_smoke failed with
    `sqlite3.OperationalError: no such column: grounding_refs` against run DBs
    created before this feature.
    """
    import aiosqlite

    db_path = str(tmp_path / "legacy.db")
    # Build a pre-feature `hypotheses` table: everything except grounding_refs.
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """CREATE TABLE hypotheses (
                   id TEXT PRIMARY KEY, run_id TEXT NOT NULL, text TEXT NOT NULL,
                   summary TEXT NOT NULL, generation_method TEXT, source TEXT,
                   category TEXT, evolved_from TEXT, elo_rating REAL DEFAULT 1200.0,
                   annotations TEXT DEFAULT '[]', status TEXT DEFAULT 'active',
                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               )"""
        )
        await db.commit()

    store = StateStore(db_path)
    await store.init_db()

    async with aiosqlite.connect(db_path) as db:
        async with db.execute("PRAGMA table_info(hypotheses)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
    assert "grounding_refs" in columns

    # And the write path that crashed now works end to end.
    await store.save_hypothesis(Hypothesis(
        id="legacy-h", run_id="run1", text="t", summary="s",
        generation_method="literature", source="system",
    ))
    await store.set_grounding_refs("legacy-h", {"28442"})
    assert await store.get_grounding_refs("legacy-h") == {"28442"}


async def test_init_db_is_idempotent_on_an_already_migrated_database(tmp_path):
    store = StateStore(str(tmp_path / "twice.db"))
    await store.init_db()
    await store.init_db()          # must not raise "duplicate column name"
    await store.save_hypothesis(Hypothesis(
        id="h", run_id="r", text="t", summary="s",
        generation_method="debate", source="system",
    ))
    assert await store.get_grounding_refs("h") is None
