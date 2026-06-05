"""Tests for the per-question reference-accuracy baseline (difficulty correction).

The reference samples N base-model responses per MCQ question and scores each
against gold; the fraction correct is that question's difficulty floor, which
feeds blue_minus_red_spread in the concordance read.
"""
import asyncio

from bench.goalset import BenchGoal
from bench.reference import reference_accuracy_for_goals, backend_generate


def _goal(gid, gold):
    return BenchGoal(id=gid, goal=f"Question {gid}? (A) x (B) y", gold_answer=gold)


def test_reference_accuracy_fraction_correct():
    """3 of 4 samples answer 'A'; gold is 'A' → reference accuracy 0.75."""
    answers = ["Answer: A", "Answer: A", "Answer: A", "Answer: B"]
    calls = {"i": 0}

    async def generate(prompt):
        i = calls["i"]; calls["i"] += 1
        return answers[i % len(answers)]

    goals = [_goal("q1", "A")]
    refs = asyncio.run(reference_accuracy_for_goals(goals, generate, n_samples=4))
    assert refs == {"q1": 0.75}


def test_reference_skips_goals_without_gold():
    """Goals lacking a gold_answer (non-MCQ) get no reference entry."""
    async def generate(prompt):
        return "Answer: A"

    goals = [_goal("q1", "A"), BenchGoal(id="q2", goal="open-ended", gold_answer=None)]
    refs = asyncio.run(reference_accuracy_for_goals(goals, generate, n_samples=2))
    assert "q1" in refs and "q2" not in refs


def test_reference_zero_samples_is_empty():
    """n_samples=0 (feature off) produces no references and makes no calls."""
    calls = {"n": 0}

    async def generate(prompt):
        calls["n"] += 1
        return "Answer: A"

    refs = asyncio.run(reference_accuracy_for_goals([_goal("q1", "A")], generate, n_samples=0))
    assert refs == {} and calls["n"] == 0


def test_reference_all_wrong_is_zero():
    """Every sample misses gold → reference accuracy 0.0 (the hardest questions)."""
    async def generate(prompt):
        return "Answer: C"

    refs = asyncio.run(reference_accuracy_for_goals([_goal("q1", "A")], generate, n_samples=3))
    assert refs == {"q1": 0.0}


def test_reference_tolerates_sample_failures():
    """A failing sample drops out; accuracy is computed over the survivors."""
    calls = {"i": 0}

    async def generate(prompt):
        i = calls["i"]; calls["i"] += 1
        if i % 2 == 0:
            raise RuntimeError("rate limit")
        return "Answer: A"  # the odd-indexed (surviving) calls are all correct

    refs = asyncio.run(reference_accuracy_for_goals([_goal("q1", "A")], generate, n_samples=4))
    # 2 of 4 raised; the 2 survivors are correct → 2/2 = 1.0 over survivors
    assert refs == {"q1": 1.0}


def test_reference_skips_goal_when_all_samples_fail():
    """A question whose every sample errors gets no reference entry (not 0.0)."""
    async def generate(prompt):
        raise RuntimeError("backend down")

    refs = asyncio.run(reference_accuracy_for_goals([_goal("q1", "A")], generate, n_samples=3))
    assert refs == {}


def test_reference_respects_concurrency_cap():
    """No more than `concurrency` samples are ever in flight at once."""
    state = {"now": 0, "peak": 0}

    async def generate(prompt):
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0)  # yield so overlap is observable
        state["now"] -= 1
        return "Answer: A"

    asyncio.run(reference_accuracy_for_goals([_goal("q1", "A")], generate,
                                             n_samples=10, concurrency=3))
    assert state["peak"] <= 3


def test_backend_generate_defaults_to_fast_and_can_override():
    """Default tier is fast (matches the Generation agent); use_strong overrides."""
    seen = {}

    class FakeBackend:
        async def call(self, system_prompt, user_prompt, *, use_strong=False, max_tokens=8192):
            seen["use_strong"] = use_strong
            seen["user_prompt"] = user_prompt
            return "Answer: A"

    assert asyncio.run(backend_generate(FakeBackend())("hi")) == "Answer: A"
    assert seen["use_strong"] is False and seen["user_prompt"] == "hi"

    asyncio.run(backend_generate(FakeBackend(), use_strong=True)("hi"))
    assert seen["use_strong"] is True
