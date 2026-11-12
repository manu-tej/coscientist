# tests/test_orchestrator.py
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock
from core.orchestrator import AgentRunner
from core.state import StateStore
from core.models import (
    Hypothesis, ResearchPlanConfig, AgentTask, AgentType, TournamentMatch,
)
from tools.evidence import EvidenceRecord, SearchOutcome


def make_outcome(*refs):
    return SearchOutcome(records=[
        EvidenceRecord(
            source="pubmed", ref=ref, title=f"Article {ref}",
            snippet="summary", url=f"https://pubmed/{ref}",
        )
        for ref in refs
    ])


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
    search.search_outcome = AsyncMock(return_value=make_outcome("111", "222"))

    runner = AgentRunner(
        store=store, config=config,
        generation=overrides.get("generation", gen),
        reflection=overrides.get("reflection", refl),
        ranking=overrides.get("ranking", ranking),
        proximity=overrides.get("proximity", proximity),
        evolution=overrides.get("evolution", evolution),
        meta_review=overrides.get("meta_review", meta),
        search=overrides.get("search", search),
        ranking_mode=overrides.get("ranking_mode", "elo"),
    )
    return runner, gen, refl, ranking, proximity, evolution, meta


async def test_generation_task_saves_hypothesis(store, config):
    runner, gen, *_ = build_runner(store, config)
    task = AgentTask(priority=1, agent_type=AgentType.GENERATION, run_id="run1", strategy="debate")
    await runner.run_task(task)
    hypotheses = await store.list_hypotheses("run1")
    assert len(hypotheses) == 1
    assert hypotheses[0].generation_method == "debate"


async def test_generation_task_activates_assigned_persona_without_mutating_run_config(
    store, config,
):
    runner, generation, *_ = build_runner(store, config)
    task = AgentTask(
        priority=1,
        agent_type=AgentType.GENERATION,
        run_id="run1",
        strategy="debate",
        extra={"persona": "disease biologist"},
    )

    await runner.run_task(task)

    task_config = generation.run_debate.await_args.args[0]
    assert task_config is not config
    assert task_config.run_id == config.run_id
    assert "Assigned specialist role: disease biologist" in task_config.goal
    assert config.goal == "ALS"


async def test_generation_literature_uses_search(store, config):
    runner, gen, *_ = build_runner(store, config)
    task = AgentTask(priority=1, agent_type=AgentType.GENERATION, run_id="run1", strategy="literature")
    await runner.run_task(task)
    gen.run_literature.assert_called_once()


async def test_generation_literature_persists_shown_refs(store, config):
    """The exact ref set shown to the model at generation time must be
    recorded, so a later citation-attribution check can diff cited-vs-shown."""
    runner, gen, *_ = build_runner(store, config)
    task = AgentTask(priority=1, agent_type=AgentType.GENERATION, run_id="run1", strategy="literature")
    await runner.run_task(task)
    hypotheses = await store.list_hypotheses("run1")
    assert len(hypotheses) == 1
    assert await store.get_grounding_refs(hypotheses[0].id) == {"111", "222"}


async def test_generation_literature_without_search_outcome_leaves_refs_unrecorded(store, config):
    """A legacy search collaborator (e.g. bench's plain SearchTool) exposes only
    search_and_format(), not search_outcome() -- generation must still succeed,
    and grounding_refs must stay unrecorded (None), not an accusable empty set."""
    search = MagicMock(spec=["search_and_format"])
    search.search_and_format = AsyncMock(return_value="[1] Article")
    runner, gen, *_ = build_runner(store, config, search=search)
    task = AgentTask(priority=1, agent_type=AgentType.GENERATION, run_id="run1", strategy="literature")
    await runner.run_task(task)
    hypotheses = await store.list_hypotheses("run1")
    assert len(hypotheses) == 1
    assert await store.get_grounding_refs(hypotheses[0].id) is None


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


async def test_reflection_claim_prevents_double_review(store, config):
    import asyncio
    runner, gen, refl, *_ = build_runner(store, config)
    h = make_h()
    await store.save_hypothesis(h)
    task = AgentTask(priority=1, agent_type=AgentType.REFLECTION, run_id="run1")
    # Fire two reflection tasks concurrently on the same (only) hypothesis
    await asyncio.gather(runner.run_task(task), runner.run_task(task))
    # Only ONE should have run the pipeline -> initial_review called exactly once
    assert refl.run_initial_review.call_count == 1


async def test_failed_reflection_releases_claim_for_retry(store, config):
    runner, gen, refl, *_ = build_runner(store, config)
    h = make_h()
    await store.save_hypothesis(h)
    refl.run_initial_review = AsyncMock(
        side_effect=[RuntimeError("provider down"), refl.run_initial_review.return_value]
    )
    task = AgentTask(
        priority=1, agent_type=AgentType.REFLECTION,
        run_id="run1", hypothesis_id=h.id,
    )

    with pytest.raises(RuntimeError, match="provider down"):
        await runner.run_task(task)
    await runner.run_task(task)

    assert refl.run_initial_review.call_count == 2


async def test_concurrent_ranking_tasks_are_serialized(store, config):
    import asyncio

    active = 0
    max_active = 0
    runner, gen, refl, ranking, *_ = build_runner(store, config)
    for _ in range(2):
        await store.save_hypothesis(make_h())

    async def run_match(h1, h2, cfg, review_1, review_2):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return TournamentMatch(
            id=str(uuid.uuid4()), run_id="run1", h1_id=h1.id, h2_id=h2.id,
            winner_id=h1.id, match_type="single_turn",
            elo_before_h1=h1.elo_rating, elo_before_h2=h2.elo_rating,
            elo_after_h1=h1.elo_rating + 16, elo_after_h2=h2.elo_rating - 16,
        )

    ranking.run_match = AsyncMock(side_effect=run_match)
    task = AgentTask(priority=1, agent_type=AgentType.RANKING, run_id="run1")
    await asyncio.gather(runner.run_task(task), runner.run_task(task))

    assert max_active == 1


async def test_absolute_ranking_never_selects_pairs_or_writes_matches(
    store, config, monkeypatch,
):
    from bench.absolute import AbsoluteRankingAgent

    class Judge:
        weights = {
            "novelty": 1.0,
            "feasibility": 1.0,
            "correctness": 1.0,
            "impact": 1.0,
        }

        def __init__(self):
            self.calls = []

        async def score(self, hypothesis, *, field):
            self.calls.append(hypothesis)
            return {
                "scores": {axis: 4 for axis in self.weights},
                "total": 4.0,
            }

    def pairwise_is_a_bug(*args, **kwargs):
        raise AssertionError("absolute mode must not select tournament pairs")

    monkeypatch.setattr("core.orchestrator.select_match_pairs", pairwise_is_a_bug)
    hypotheses = [make_h() for _ in range(3)]
    for hypothesis in hypotheses:
        await store.save_hypothesis(hypothesis)
    judge = Judge()
    absolute = AbsoluteRankingAgent(
        judge=judge, store=store, scorer_id="absolute-test"
    )
    runner, *_ = build_runner(
        store, config, ranking=absolute, ranking_mode="absolute"
    )
    task = AgentTask(priority=1, agent_type=AgentType.RANKING, run_id="run1")

    await runner.run_task(task)
    await runner.run_task(task)

    assert len(judge.calls) == 3
    assert await store.list_matches("run1") == []
    assert all(
        hypothesis.elo_rating == 1200.0
        for hypothesis in await store.list_hypotheses("run1")
    )


async def test_absolute_ranking_scores_new_hypotheses_incrementally(store, config):
    from bench.absolute import AbsoluteRankingAgent

    judge = MagicMock()
    judge.weights = {
        "novelty": 1.0,
        "feasibility": 1.0,
        "correctness": 1.0,
        "impact": 1.0,
    }
    judge.score = AsyncMock(return_value={
        "scores": {axis: 3 for axis in judge.weights},
        "total": 3.0,
    })
    absolute = AbsoluteRankingAgent(
        judge=judge, store=store, scorer_id="absolute-test"
    )
    runner, *_ = build_runner(
        store, config, ranking=absolute, ranking_mode="absolute"
    )
    first = make_h()
    second = make_h()
    await store.save_hypothesis(first)
    task = AgentTask(priority=1, agent_type=AgentType.RANKING, run_id="run1")

    await runner.run_task(task)
    await runner.run_task(task)
    await store.save_hypothesis(second)
    await runner.finalize_ranking()

    assert judge.score.await_count == 2
    assert len(await store.list_absolute_scores("run1")) == 2
