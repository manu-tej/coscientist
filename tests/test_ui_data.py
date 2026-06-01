# tests/test_ui_data.py
import pytest
import uuid
from core.state import StateStore
from core.models import Hypothesis, Review
from ui.data import (
    get_ranked_hypotheses, get_research_overview, inject_expert_hypothesis,
    submit_expert_review,
)


@pytest.fixture
async def store(tmp_path):
    s = StateStore(str(tmp_path / "t.db"))
    await s.init_db()
    return s


def make_h(elo, summary="summary"):
    return Hypothesis(
        id=str(uuid.uuid4()), run_id="run1", text="full text here",
        summary=summary, generation_method="debate", source="system", elo_rating=elo,
    )


async def test_ranked_hypotheses_sorted_by_elo(store):
    await store.save_hypothesis(make_h(1200.0, "low"))
    await store.save_hypothesis(make_h(1500.0, "high"))
    await store.save_hypothesis(make_h(1350.0, "mid"))
    rows = await get_ranked_hypotheses(store, "run1")
    assert rows[0]["summary"] == "high"
    assert rows[0]["elo"] == 1500.0
    assert rows[-1]["summary"] == "low"


async def test_ranked_hypotheses_excludes_rejected(store):
    h = make_h(1200.0)
    await store.save_hypothesis(h)
    await store.set_hypothesis_status(h.id, "rejected")
    rows = await get_ranked_hypotheses(store, "run1")
    assert len(rows) == 0


async def test_research_overview_returns_latest(store):
    await store.save_meta_review(
        id="m1", run_id="run1", meta_critique="c",
        research_overview="OVERVIEW A", research_contacts="contacts", tick=1,
    )
    await store.save_meta_review(
        id="m2", run_id="run1", meta_critique="c",
        research_overview="OVERVIEW B", research_contacts="contacts", tick=2,
    )
    overview = await get_research_overview(store, "run1")
    assert overview == "OVERVIEW B"


async def test_research_overview_empty(store):
    overview = await get_research_overview(store, "run1")
    assert "No research overview" in overview


async def test_inject_expert_hypothesis(store):
    h_id = await inject_expert_hypothesis(store, "run1", "My hypothesis: X causes Y")
    stored = await store.get_hypothesis(h_id)
    assert stored.source == "expert"
    assert stored.elo_rating == 1200.0
    assert "X causes Y" in stored.text


async def test_submit_expert_review(store):
    h = make_h(1200.0)
    await store.save_hypothesis(h)
    await submit_expert_review(store, h.id, "This needs more controls.")
    reviews = await store.list_reviews(h.id)
    assert len(reviews) == 1
    assert reviews[0].tier == 0  # expert review
    assert "controls" in reviews[0].critique
