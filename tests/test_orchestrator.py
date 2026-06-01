# tests/test_orchestrator.py
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock
from core.orchestrator import AgentRunner
from core.state import StateStore
from core.models import (
    Hypothesis, ResearchPlanConfig, AgentTask, AgentType, TournamentMatch,
)


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


def make_h(method="debate", elo=1200.0, source="system"):
    return Hypothesis(
        id=str(uuid.uuid4()), run_id="run1", text="text", summary="summary",
        generation_method=method, source=source, elo_rating=elo,
    )


def build_runner(store, config, **overrides):
    gen = MagicMock()
    gen.run_literature = AsyncMock(return_value=make_h("literature"))
    gen.run_debate = AsyncMock(return_value=make_h("debate"))
    gen.run_assumptions = AsyncMock(return_value=make_h("assumptions"))
    gen.run_expansion = AsyncMock(return_value=make_h("expansion"))

    refl = MagicMock()
    from core.models import Review
    refl.run_initial_review = AsyncMock(return_value=Review(id="r", hypothesis_id="x", tier=1, critique="ok", verdict="passed"))
    refl.run_full_review = AsyncMock(return_value=Review(id="r", hypothesis_id="x", tier=2, critique="ok", verdict="passed"))
    refl.run_deep_verification = AsyncMock(return_value=Review(id="r", hypothesis_id="x", tier=3, critique="ok", verdict="passed"))
    refl.run_observation_review = AsyncMock(return_value=(Review(id="r", hypothesis_id="x", tier=4, critique="ok", verdict="passed"), None))
    refl.run_simulation_review = AsyncMock(return_value=Review(id="r", hypothesis_id="x", tier=5, critique="ok", verdict="passed"))
    refl.run_tournament_review = AsyncMock(return_value=Review(id="r", hypothesis_id="x", tier=6, critique="ok", verdict="passed"))

    ranking = MagicMock()
    ranking.run_match = AsyncMock(return_value=TournamentMatch(
        id=str(uuid.uuid4()), run_id="run1", h1_id="a", h2_id="b",
        winner_id="a", match_type="single_turn",
        elo_before_h1=1200.0, elo_before_h2=1200.0,
        elo_after_h1=1216.0, elo_after_h2=1184.0,
    ))

    proximity = MagicMock()
    proximity.update_graph = AsyncMock(return_value=[])

    evolution = MagicMock()
    evolution.run_grounding = AsyncMock(return_value=make_h("grounding"))
    evolution.run_coherence = AsyncMock(return_value=make_h("coherence"))
    evolution.run_inspiration = AsyncMock(return_value=make_h("inspiration"))
    evolution.run_combination = AsyncMock(return_value=make_h("combination"))
    evolution.run_simplification = AsyncMock(return_value=make_h("simplification"))
    evolution.run_out_of_box = AsyncMock(return_value=make_h("out_of_box"))

    meta = MagicMock()
    meta.run_meta_critique = AsyncMock(return_value="META CRITIQUE TEXT")
    meta.run_research_overview = AsyncMock(return_value="OVERVIEW TEXT")
    meta.run_research_contacts = AsyncMock(return_value="CONTACTS TEXT")

    search = MagicMock()
    search.search_and_format = AsyncMock(return_value="[1] Article")

    runner = AgentRunner(
        store=store, config=config,
        generation=overrides.get("generation", gen),
        reflection=overrides.get("reflection", refl),
        ranking=overrides.get("ranking", ranking),
        proximity=overrides.get("proximity", proximity),
        evolution=overrides.get("evolution", evolution),
        meta_review=overrides.get("meta_review", meta),
        search=overrides.get("search", search),
    )
    return runner, gen, refl, ranking, proximity, evolution, meta


async def test_generation_task_saves_hypothesis(store, config):
    runner, gen, *_ = build_runner(store, config)
    task = AgentTask(priority=1, agent_type=AgentType.GENERATION, run_id="run1", strategy="debate")
    await runner.run_task(task)
    hypotheses = await store.list_hypotheses("run1")
    assert len(hypotheses) == 1
    assert hypotheses[0].generation_method == "debate"


async def test_generation_literature_uses_search(store, config):
    runner, gen, *_ = build_runner(store, config)
    task = AgentTask(priority=1, agent_type=AgentType.GENERATION, run_id="run1", strategy="literature")
    await runner.run_task(task)
    gen.run_literature.assert_called_once()


async def test_reflection_task_reviews_pending(store, config):
    runner, gen, refl, *_ = build_runner(store, config)
    h = make_h()
    await store.save_hypothesis(h)
    task = AgentTask(priority=1, agent_type=AgentType.REFLECTION, run_id="run1", hypothesis_id=h.id)
    await runner.run_task(task)
    refl.run_initial_review.assert_called_once()
    reviews = await store.list_reviews(h.id)
    assert len(reviews) >= 1


async def test_ranking_task_saves_match_and_updates_elo(store, config):
    runner, gen, refl, ranking, *_ = build_runner(store, config)
    h1 = make_h()
    h2 = make_h()
    await store.save_hypothesis(h1)
    await store.save_hypothesis(h2)
    from core.models import Review
    await store.save_review(Review(id=str(uuid.uuid4()), hypothesis_id=h1.id, tier=1, critique="r1", verdict="passed"))
    await store.save_review(Review(id=str(uuid.uuid4()), hypothesis_id=h2.id, tier=1, critique="r2", verdict="passed"))
    ranking.run_match = AsyncMock(return_value=TournamentMatch(
        id=str(uuid.uuid4()), run_id="run1", h1_id=h1.id, h2_id=h2.id,
        winner_id=h1.id, match_type="single_turn",
        elo_before_h1=1200.0, elo_before_h2=1200.0,
        elo_after_h1=1216.0, elo_after_h2=1184.0,
    ))
    task = AgentTask(priority=1, agent_type=AgentType.RANKING, run_id="run1")
    await runner.run_task(task)
    matches = await store.list_matches("run1")
    assert len(matches) == 1


async def test_evolution_task_creates_new_hypothesis(store, config):
    runner, gen, refl, ranking, proximity, evolution, meta = build_runner(store, config)
    h = make_h(elo=1400.0)
    await store.save_hypothesis(h)
    task = AgentTask(priority=1, agent_type=AgentType.EVOLUTION, run_id="run1", strategy="simplification")
    await runner.run_task(task)
    evolution.run_simplification.assert_called_once()
    hypotheses = await store.list_hypotheses("run1")
    assert len(hypotheses) == 2  # original + evolved


async def test_meta_review_task_saves_meta_review(store, config):
    runner, gen, refl, ranking, proximity, evolution, meta = build_runner(store, config)
    h = make_h()
    await store.save_hypothesis(h)
    from core.models import Review
    await store.save_review(Review(id=str(uuid.uuid4()), hypothesis_id=h.id, tier=1, critique="some critique", verdict="passed"))
    task = AgentTask(priority=1, agent_type=AgentType.META_REVIEW, run_id="run1", extra={"tick": 5})
    await runner.run_task(task)
    meta.run_meta_critique.assert_called_once()
    latest = await store.get_latest_meta_review("run1")
    assert latest is not None
    assert latest["meta_critique"] == "META CRITIQUE TEXT"


async def test_proximity_task_updates_graph(store, config):
    runner, gen, refl, ranking, proximity, *_ = build_runner(store, config)
    await store.save_hypothesis(make_h())
    await store.save_hypothesis(make_h())
    task = AgentTask(priority=1, agent_type=AgentType.PROXIMITY, run_id="run1")
    await runner.run_task(task)
    proximity.update_graph.assert_called_once()
