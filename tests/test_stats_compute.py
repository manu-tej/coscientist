# tests/test_stats_compute.py
import pytest
import uuid
from core.stats import compute_stats
from core.state import StateStore
from core.models import AbsoluteScore, Hypothesis, Review


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


async def test_compute_stats_uses_native_absolute_scale_and_tracks_unranked(store):
    high = make_h(900.0, method="debate")
    low = make_h(2000.0, method="combination")
    pending = make_h(5000.0, method="debate")
    for hypothesis in (high, low, pending):
        await store.save_hypothesis(hypothesis)
    for hypothesis, total in ((high, 5.0), (low, 2.0)):
        claim = await store.claim_absolute_score(hypothesis.id, "absolute-test")
        value = int(total)
        await store.publish_absolute_score(AbsoluteScore(
            run_id="run1",
            hypothesis_id=hypothesis.id,
            scorer_id="absolute-test",
            axis_scores={
                "novelty": value,
                "feasibility": value,
                "correctness": value,
                "impact": value,
            },
            total=total,
        ), owner_token=claim.owner_token)

    stats = await compute_stats(
        store,
        "run1",
        last_meta_review_tick=0,
        current_tick=0,
        ranking_mode="absolute",
    )

    assert stats.ranking_mode == "absolute"
    assert stats.ranking_variance == 2.25
    assert stats.n_unranked == 1
    assert stats.tournament_progress == 2.0
    assert stats.generation_effectiveness == 5.0
    assert stats.evolution_effectiveness == 2.0


async def test_absolute_stats_ignore_scores_for_rejected_hypotheses(store):
    active_one = make_h(1200.0)
    active_two = make_h(1200.0)
    rejected = make_h(1200.0)
    for hypothesis in (active_one, active_two, rejected):
        await store.save_hypothesis(hypothesis)
    await store.set_hypothesis_status(rejected.id, "rejected")
    await store.save_proximity(active_one.id, active_two.id, 0.2)
    await store.save_proximity(active_one.id, rejected.id, 0.99)

    for hypothesis, value in (
        (active_one, 3),
        (active_two, 3),
        (rejected, 5),
    ):
        claim = await store.claim_absolute_score(
            hypothesis.id, "absolute-test"
        )
        await store.publish_absolute_score(AbsoluteScore(
            run_id="run1",
            hypothesis_id=hypothesis.id,
            scorer_id="absolute-test",
            axis_scores={
                "novelty": value,
                "feasibility": value,
                "correctness": value,
                "impact": value,
            },
            total=float(value),
        ), owner_token=claim.owner_token)

    stats = await compute_stats(
        store,
        "run1",
        last_meta_review_tick=0,
        current_tick=0,
        ranking_mode="absolute",
    )

    assert stats.n_hypotheses == 2
    assert stats.ranking_variance == 0.0
    assert stats.tournament_progress == 2.0
    assert stats.n_unranked == 0
    assert stats.avg_proximity == 0.2
