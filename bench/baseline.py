from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from bench.goalset import BenchGoal
from bench.runner import BenchHypothesis

SINGLE_SHOT_PROMPT = (
    "Research goal: {goal}\n\n"
    "Propose your single best novel, testable hypothesis. Include a clear "
    "rationale and a concrete validation plan. If the goal is a multiple-choice "
    "question, state the chosen option as 'Answer: <letter>' and justify it."
)

GenerateFn = Callable[[str], Awaitable[str]]
JudgeScoreFn = Callable[[str], Awaitable[float]]


def _bh(goal: BenchGoal, idx: int, text: str, variant: str) -> BenchHypothesis:
    return BenchHypothesis(
        id=f"{variant}-{goal.id}-{idx}", text=text, summary=text[:80],
        elo_rating=1200.0, created_at="t0", generation_method=variant,
    )


async def single_shot(goal: BenchGoal, generate: GenerateFn) -> list[BenchHypothesis]:
    """One generator call → one hypothesis (the n=1 floor control)."""
    text = await generate(SINGLE_SHOT_PROMPT.format(goal=goal.goal))
    return [_bh(goal, 0, text, "single_shot")]


async def best_of_n(
    goal: BenchGoal, generate: GenerateFn, judge_score: JudgeScoreFn, n: int = 32,
) -> list[BenchHypothesis]:
    """Sample n single-shot hypotheses; keep the best by judge score.
    Controls for 'the gain is just more sampling' (§9)."""
    prompt = SINGLE_SHOT_PROMPT.format(goal=goal.goal)
    texts = await asyncio.gather(*(generate(prompt) for _ in range(n)))
    scores = await asyncio.gather(*(judge_score(t) for t in texts))
    best_idx = max(range(n), key=lambda i: scores[i])
    return [_bh(goal, best_idx, texts[best_idx], "best_of_32")]


async def all_of_n(
    goal: BenchGoal, generate: GenerateFn, n: int = 32,
) -> list[BenchHypothesis]:
    """Return all n samples (used as the concordance reference baseline: the 32
    base-model samples per question, §6 step 5). Reused by Tier-3 best_of_n above."""
    prompt = SINGLE_SHOT_PROMPT.format(goal=goal.goal)
    texts = await asyncio.gather(*(generate(prompt) for _ in range(n)))
    return [_bh(goal, i, t, "reference") for i, t in enumerate(texts)]
