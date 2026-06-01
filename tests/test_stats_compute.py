# tests/test_stats_compute.py
import pytest
import uuid
from core.stats import compute_stats
from core.state import StateStore
from core.models import Hypothesis, Review


@pytest.fixture
async def store(tmp_path):
    s = StateStore(str(tmp_path / "t.db"))
    await s.init_db()
    return s


def make_h(elo, method="debate"):
    return Hypothesis(
        id=str(uuid.uuid4()), run_id="run1", text="t", summary="s",
        generation_method=method, source="system", elo_rating=elo,
    )


async def test_compute_stats_counts_hypotheses(store):
    for elo in [1200.0, 1300.0, 1400.0]:
        await store.save_hypothesis(make_h(elo))
    stats = await compute_stats(store, "run1", last_meta_review_tick=0, current_tick=5)
    assert stats.n_hypotheses == 3
    assert stats.last_meta_review_age == 5


async def test_compute_stats_elo_variance(store):
    await store.save_hypothesis(make_h(1200.0))
    await store.save_hypothesis(make_h(1200.0))
    stats = await compute_stats(store, "run1", last_meta_review_tick=0, current_tick=0)
    assert stats.elo_variance == 0.0  # identical ratings -> zero variance


async def test_compute_stats_pending_review(store):
    h1 = make_h(1200.0)
    h2 = make_h(1200.0)
    await store.save_hypothesis(h1)
    await store.save_hypothesis(h2)
    # h1 has a tier-1 review, h2 does not
    await store.save_review(Review(id="r1", hypothesis_id=h1.id, tier=1, critique="ok", verdict="passed"))
    stats = await compute_stats(store, "run1", last_meta_review_tick=0, current_tick=0)
    assert stats.n_reviewed == 1
    assert stats.n_pending_review == 1


async def test_compute_stats_effectiveness_by_method(store):
    await store.save_hypothesis(make_h(1400.0, method="debate"))       # generation
    await store.save_hypothesis(make_h(1200.0, method="combination"))  # evolution
    stats = await compute_stats(store, "run1", last_meta_review_tick=0, current_tick=0)
    assert stats.generation_effectiveness > stats.evolution_effectiveness
