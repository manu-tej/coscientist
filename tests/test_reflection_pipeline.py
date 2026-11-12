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


from tools.evidence import ClaimVerdict


class _Verifier:
    def __init__(self, *verdicts, raises=False):
        self.verdicts, self.raises = list(verdicts), raises

    async def verify(self, hypothesis_text, sources, backend):
        if self.raises:
            raise RuntimeError("verifier exploded")
        return self.verdicts


async def test_contradicted_claim_quarantines_the_hypothesis(store, config, reflection, search):
    h = make_h()
    await store.save_hypothesis(h)

    verdict = await run_reflection_pipeline(
        h, config, reflection, search, store,
        verifier=_Verifier(ClaimVerdict(claim="c", status="contradicted",
                                        rationale="the study reports the opposite")),
    )

    assert verdict == "flagged"
    stored = await store.get_hypothesis(h.id)
    assert stored.status == "quarantined"
    assert [v.status for v in await store.list_claim_verdicts(h.id)] == ["contradicted"]


async def test_supported_claims_do_not_override_the_model(store, config, reflection, search):
    h = make_h()
    await store.save_hypothesis(h)

    verdict = await run_reflection_pipeline(
        h, config, reflection, search, store,
        verifier=_Verifier(ClaimVerdict(claim="c", status="supported")),
    )

    assert verdict == "passed"
    assert len(await store.list_claim_verdicts(h.id)) == 1


async def test_no_verifier_preserves_existing_behaviour(store, config, reflection, search):
    h = make_h()
    await store.save_hypothesis(h)
    assert await run_reflection_pipeline(h, config, reflection, search, store) == "passed"
    assert await store.list_claim_verdicts(h.id) == []


async def test_verifier_failure_records_a_sentinel_and_never_fails_the_review(
    store, config, reflection, search,
):
    """Zero rows must never be ambiguous between 'nothing checkable' and 'down'."""
    h = make_h()
    await store.save_hypothesis(h)

    verdict = await run_reflection_pipeline(
        h, config, reflection, search, store, verifier=_Verifier(raises=True),
    )

    assert verdict == "passed"
    stored = await store.list_claim_verdicts(h.id)
    assert [v.status for v in stored] == ["unchecked"]
    assert "verifier raised" in stored[0].rationale


async def test_no_claims_records_a_sentinel(store, config, reflection, search):
    h = make_h()
    await store.save_hypothesis(h)
    await run_reflection_pipeline(h, config, reflection, search, store,
                                  verifier=_Verifier())
    stored = await store.list_claim_verdicts(h.id)
    assert [v.status for v in stored] == ["unchecked"]
    assert "no checkable claims" in stored[0].rationale


async def test_save_claim_verdict_failure_does_not_raise_and_pipeline_continues(
    store, config, reflection, search, monkeypatch,
):
    """A store failure while persisting verdicts must degrade, not raise into
    the supervisor -- the run continues and produces a normal verdict."""
    h = make_h()
    await store.save_hypothesis(h)

    async def _raise(*args, **kwargs):
        raise RuntimeError("db write exploded")

    monkeypatch.setattr(store, "save_claim_verdict", _raise)

    verdict = await run_reflection_pipeline(
        h, config, reflection, search, store,
        verifier=_Verifier(ClaimVerdict(claim="c", status="supported")),
    )

    assert verdict == "passed"
    reflection.run_deep_verification.assert_called_once()


async def test_fabricated_citation_quarantines_the_hypothesis(store, config, reflection, search):
    """A hypothesis citing a PMID never shown to it at generation time must be
    caught even when the model's own claims all check out -- this is the
    citation-attribution check, separate from claim substance verification."""
    h = Hypothesis(
        id=str(uuid.uuid4()), run_id="run1",
        text="The mechanism is well established (PMID:99999999).",
        summary="summary", generation_method="literature", source="system",
    )
    await store.save_hypothesis(h)
    await store.set_grounding_refs(h.id, {"28442"})

    verdict = await run_reflection_pipeline(
        h, config, reflection, search, store, verifier=_Verifier(),
    )

    assert verdict == "flagged"
    stored = await store.get_hypothesis(h.id)
    assert stored.status == "quarantined"
    statuses = [v.status for v in await store.list_claim_verdicts(h.id)]
    assert "contradicted" in statuses
    verdicts = await store.list_claim_verdicts(h.id)
    fabricated = [v for v in verdicts if v.status == "contradicted"]
    assert fabricated[0].claim == "citation PMID 99999999"
    assert "never retrieved" in fabricated[0].rationale


async def test_citation_matching_a_shown_ref_does_not_quarantine(store, config, reflection, search):
    h = Hypothesis(
        id=str(uuid.uuid4()), run_id="run1",
        text="The mechanism is well established (PMID:28442).",
        summary="summary", generation_method="literature", source="system",
    )
    await store.save_hypothesis(h)
    await store.set_grounding_refs(h.id, {"28442"})

    verdict = await run_reflection_pipeline(
        h, config, reflection, search, store, verifier=_Verifier(),
    )

    assert verdict == "passed"
    stored = await store.get_hypothesis(h.id)
    assert stored.status == "active"


async def test_citation_check_is_a_noop_when_grounding_was_never_recorded(
    store, config, reflection, search,
):
    """A hypothesis with no recorded grounding set (e.g. generated by a
    non-literature strategy, or a legacy search collaborator that never called
    set_grounding_refs) must not be accused of fabrication -- absence of a
    record is not evidence of invention."""
    h = Hypothesis(
        id=str(uuid.uuid4()), run_id="run1",
        text="The mechanism is well established (PMID:99999999).",
        summary="summary", generation_method="debate", source="system",
    )
    await store.save_hypothesis(h)

    verdict = await run_reflection_pipeline(
        h, config, reflection, search, store, verifier=_Verifier(),
    )

    assert verdict == "passed"
    verdicts = await store.list_claim_verdicts(h.id)
    assert not any(v.status == "contradicted" for v in verdicts)


async def test_set_hypothesis_status_failure_on_contradiction_does_not_flag(
    store, config, reflection, search, monkeypatch,
):
    """If quarantining fails, the pipeline must not claim "flagged" -- the
    hypothesis was never actually quarantined, so the honest outcome is to
    fall through to tier 3 rather than assert a status change that didn't
    durably happen."""
    h = make_h()
    await store.save_hypothesis(h)

    async def _raise(*args, **kwargs):
        raise RuntimeError("status write exploded")

    monkeypatch.setattr(store, "set_hypothesis_status", _raise)

    verdict = await run_reflection_pipeline(
        h, config, reflection, search, store,
        verifier=_Verifier(ClaimVerdict(claim="c", status="contradicted",
                                        rationale="the study reports the opposite")),
    )

    assert verdict == "passed"
    reflection.run_deep_verification.assert_called_once()
