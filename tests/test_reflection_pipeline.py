import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock
from core.reflection_pipeline import run_reflection_pipeline
from core.state import StateStore
from core.models import Hypothesis, Review, ResearchPlanConfig


@pytest.fixture
async def store(tmp_path):
    s = StateStore(str(tmp_path / "t.db"))
    await s.init_db()
    return s


@pytest.fixture
def config():
    return ResearchPlanConfig(
        run_id="run1", goal="ALS", preferences="Novel",
        attributes=["Novelty"], constraints="testable", safety_approved=True,
    )


def make_h():
    return Hypothesis(
        id=str(uuid.uuid4()), run_id="run1", text="hypothesis text",
        summary="summary", generation_method="debate", source="system",
    )


def passing_review(tier):
    return Review(id=str(uuid.uuid4()), hypothesis_id="x", tier=tier, critique="ok", verdict="passed")


@pytest.fixture
def reflection():
    r = MagicMock()
    r.run_initial_review = AsyncMock(return_value=passing_review(1))
    r.run_full_review = AsyncMock(return_value=passing_review(2))
    r.run_deep_verification = AsyncMock(return_value=passing_review(3))
    r.run_observation_review = AsyncMock(return_value=(passing_review(4), "missing piece: TDP-43"))
    r.run_simulation_review = AsyncMock(return_value=passing_review(5))
    r.run_tournament_review = AsyncMock(return_value=passing_review(6))
    return r


@pytest.fixture
def search():
    s = MagicMock()
    s.search_and_format = AsyncMock(return_value="[1] Article A\nSummary: ...")
    return s


async def test_pipeline_early_exit_on_rejection(store, config, reflection, search):
    h = make_h()
    await store.save_hypothesis(h)
    # tier 1 rejects
    rejected = Review(id=str(uuid.uuid4()), hypothesis_id=h.id, tier=1, critique="bad", verdict="rejected")
    reflection.run_initial_review = AsyncMock(return_value=rejected)
    verdict = await run_reflection_pipeline(h, config, reflection, search, store)
    assert verdict == "rejected"
    stored = await store.get_hypothesis(h.id)
    assert stored.status == "rejected"
    reflection.run_full_review.assert_not_called()


async def test_pipeline_runs_all_tiers_on_pass(store, config, reflection, search):
    h = make_h()
    await store.save_hypothesis(h)
    verdict = await run_reflection_pipeline(h, config, reflection, search, store, meta_critique="prior critique")
    assert verdict == "passed"
    reflection.run_initial_review.assert_called_once()
    reflection.run_full_review.assert_called_once()
    reflection.run_deep_verification.assert_called_once()
    reflection.run_observation_review.assert_called_once()
    reflection.run_simulation_review.assert_called_once()
    # tier 6 only runs if hypothesis has tournament history -> skipped here
    reflection.run_tournament_review.assert_not_called()


async def test_pipeline_persists_reviews(store, config, reflection, search):
    h = make_h()
    await store.save_hypothesis(h)
    await run_reflection_pipeline(h, config, reflection, search, store)
    reviews = await store.list_reviews(h.id)
    tiers = {r.tier for r in reviews}
    assert {1, 2, 3, 4, 5}.issubset(tiers)


async def test_pipeline_appends_observation(store, config, reflection, search):
    h = make_h()
    await store.save_hypothesis(h)
    await run_reflection_pipeline(h, config, reflection, search, store)
    stored = await store.get_hypothesis(h.id)
    assert any("TDP-43" in a for a in stored.annotations)


async def test_pipeline_runs_tier6_with_tournament_history(store, config, reflection, search):
    h = make_h()
    await store.save_hypothesis(h)
    from core.models import TournamentMatch
    await store.save_match(TournamentMatch(
        id=str(uuid.uuid4()), run_id="run1", h1_id=h.id, h2_id="other",
        winner_id=h.id, match_type="single_turn",
    ))
    await run_reflection_pipeline(h, config, reflection, search, store, meta_critique="critique")
    reflection.run_tournament_review.assert_called_once()
