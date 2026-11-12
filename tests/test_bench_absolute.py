import asyncio
from collections import Counter

import pytest

from bench.absolute import AbsoluteRankingAgent
from bench.errors import BenchError
from core.models import Hypothesis, ResearchPlanConfig
from core.state import StateStore


def _config() -> ResearchPlanConfig:
    return ResearchPlanConfig(
        run_id="run-absolute",
        goal="discover a testable mechanism",
        preferences="novel",
        attributes=["Novelty", "Feasibility"],
        constraints="testable",
        safety_approved=True,
    )


def _hypothesis(identifier: str, text: str) -> Hypothesis:
    return Hypothesis(
        id=identifier,
        run_id="run-absolute",
        text=text,
        summary=text,
        generation_method="debate",
        source="system",
    )


def _result(score: int) -> dict:
    return {
        "scores": {
            "novelty": score,
            "feasibility": score,
            "correctness": score,
            "impact": score,
        },
        "total": float(score),
    }


class _Judge:
    weights = {
        "novelty": 1.0,
        "feasibility": 1.0,
        "correctness": 1.0,
        "impact": 1.0,
    }

    def __init__(self, scores, *, delay=0.0):
        self.scores = scores
        self.delay = delay
        self.calls = []

    async def score(self, hypothesis, *, field):
        self.calls.append((hypothesis, field))
        if self.delay:
            await asyncio.sleep(self.delay)
        value = self.scores[hypothesis]
        if isinstance(value, BaseException):
            raise value
        return value


@pytest.fixture
async def absolute_store(tmp_path):
    store = StateStore(str(tmp_path / "absolute.db"))
    await store.init_db()
    await store.save_config(_config())
    return store


async def test_absolute_ranking_scores_each_hypothesis_once_and_keeps_elo(
    absolute_store,
):
    hypotheses = [
        _hypothesis("h-low", "low"),
        _hypothesis("h-high", "high"),
    ]
    for hypothesis in hypotheses:
        await absolute_store.save_hypothesis(hypothesis)
    judge = _Judge({"low": _result(2), "high": _result(5)})
    ranker = AbsoluteRankingAgent(
        judge=judge,
        store=absolute_store,
        scorer_id="absolute-test",
        max_concurrency=2,
    )

    await ranker.score_hypotheses(hypotheses, _config())
    await ranker.score_hypotheses(hypotheses, _config())
    ordered = await ranker.order_hypotheses(hypotheses)

    assert Counter(text for text, _ in judge.calls) == {"low": 1, "high": 1}
    assert [hypothesis.id for hypothesis in ordered] == ["h-high", "h-low"]
    assert await absolute_store.list_matches("run-absolute") == []
    persisted = [
        await absolute_store.get_hypothesis(hypothesis.id)
        for hypothesis in hypotheses
    ]
    assert all(hypothesis.elo_rating == 1200.0 for hypothesis in persisted)


async def test_absolute_scoring_is_single_flight_across_runner_instances(
    absolute_store,
):
    hypotheses = [_hypothesis(f"h-{index}", f"candidate-{index}") for index in range(3)]
    for hypothesis in hypotheses:
        await absolute_store.save_hypothesis(hypothesis)
    judge = _Judge(
        {hypothesis.text: _result(4) for hypothesis in hypotheses},
        delay=0.25,
    )
    rankers = [
        AbsoluteRankingAgent(
            judge=judge,
            store=StateStore(absolute_store.db_path),
            scorer_id="absolute-test",
            max_concurrency=3,
            claim_lease_seconds=0.1,
            claim_poll_seconds=0.01,
        )
        for _ in range(2)
    ]

    await asyncio.gather(*(
        ranker.score_hypotheses(hypotheses, _config()) for ranker in rankers
    ))

    assert Counter(text for text, _ in judge.calls) == {
        hypothesis.text: 1 for hypothesis in hypotheses
    }
    assert len(await absolute_store.list_absolute_scores("run-absolute")) == 3


async def test_absolute_failure_releases_claim_for_retry(absolute_store):
    hypothesis = _hypothesis("h-retry", "retry")
    await absolute_store.save_hypothesis(hypothesis)
    judge = _Judge({"retry": {"scores": {"novelty": 4}, "total": 4.0}})
    ranker = AbsoluteRankingAgent(
        judge=judge, store=absolute_store, scorer_id="absolute-test"
    )

    with pytest.raises(BenchError, match="rubric"):
        await ranker.score_hypotheses([hypothesis], _config())
    assert await absolute_store.get_absolute_score(hypothesis.id) is None

    judge.scores["retry"] = _result(4)
    await ranker.score_hypotheses([hypothesis], _config())
    assert (await absolute_store.get_absolute_score(hypothesis.id)).total == 4.0
    assert len(judge.calls) == 2


async def test_absolute_batch_failure_cancels_and_drains_sibling_scores(
    absolute_store,
):
    started = asyncio.Event()

    class Judge:
        weights = _Judge.weights

        async def score(self, hypothesis, *, field):
            if hypothesis == "invalid":
                await started.wait()
                return {"scores": {"novelty": 4}, "total": 4.0}
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                # Simulate a custom provider that ignores cancellation briefly.
                await asyncio.sleep(0.05)
                return _result(5)

    hypotheses = [
        _hypothesis("h-invalid", "invalid"),
        _hypothesis("h-slow", "slow"),
    ]
    for hypothesis in hypotheses:
        await absolute_store.save_hypothesis(hypothesis)
    ranker = AbsoluteRankingAgent(
        judge=Judge(),
        store=absolute_store,
        scorer_id="absolute-test",
        max_concurrency=2,
    )

    with pytest.raises(BenchError):
        await ranker.score_hypotheses(hypotheses, _config())

    assert await absolute_store.get_absolute_score("h-invalid") is None
    assert await absolute_store.get_absolute_score("h-slow") is None
    await asyncio.sleep(0.06)
    assert await absolute_store.get_absolute_score("h-slow") is None


@pytest.mark.parametrize(
    "scores,total",
    [
        ({"novelty": True, "feasibility": 3, "correctness": 3, "impact": 3}, 3.0),
        ({"novelty": 3.5, "feasibility": 3, "correctness": 3, "impact": 3}, 3.0),
        ({"novelty": 0, "feasibility": 3, "correctness": 3, "impact": 3}, 2.25),
        ({"novelty": 6, "feasibility": 3, "correctness": 3, "impact": 3}, 3.75),
        ({"novelty": 3, "feasibility": 3, "correctness": 3, "impact": 3}, float("nan")),
    ],
)
async def test_absolute_ranking_rejects_unsafe_score_values(
    absolute_store, scores, total,
):
    hypothesis = _hypothesis("h-invalid", "invalid")
    await absolute_store.save_hypothesis(hypothesis)
    judge = _Judge({"invalid": {"scores": scores, "total": total}})
    ranker = AbsoluteRankingAgent(
        judge=judge, store=absolute_store, scorer_id="absolute-test"
    )

    with pytest.raises(BenchError):
        await ranker.score_hypotheses([hypothesis], _config())
    assert await absolute_store.get_absolute_score(hypothesis.id) is None
