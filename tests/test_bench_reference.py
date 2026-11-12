import pytest

from bench.goalset import BenchGoal
from bench.reference import ReferenceSamplingError, reference_accuracy_for_goals


@pytest.mark.asyncio
async def test_reference_sampling_fails_closed_on_partial_responses():
    calls = 0

    async def mostly_broken(prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "Answer: B"
        raise RuntimeError("rate limited")

    goal = BenchGoal(id="q1", goal="question", gold_answer="B")
    with pytest.raises(ReferenceSamplingError, match="1/4"):
        await reference_accuracy_for_goals(
            [goal], mostly_broken, n_samples=4, concurrency=2,
        )


@pytest.mark.asyncio
async def test_reference_sampling_reports_exact_complete_accuracy():
    replies = iter(["Answer: B", "Answer: A", "Answer: B", "Answer: C"])

    async def generate(prompt):
        return next(replies)

    goal = BenchGoal(id="q1", goal="question", gold_answer="B")
    refs = await reference_accuracy_for_goals(
        [goal], generate, n_samples=4, concurrency=1,
    )
    assert refs == {"q1": 0.5}


@pytest.mark.asyncio
async def test_reference_sampling_rejects_zero_concurrency():
    goal = BenchGoal(id="q1", goal="question", gold_answer="B")
    with pytest.raises(ValueError, match="concurrency"):
        await reference_accuracy_for_goals(
            [goal], lambda prompt: None, n_samples=1, concurrency=0,
        )
