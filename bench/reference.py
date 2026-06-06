"""Per-question reference accuracy — the paper's difficulty-correction baseline.

Pooled Elo↔correctness concordance is confounded by question difficulty: an easy
question contributes mostly-correct hypotheses at every Elo, a hard one
mostly-wrong, so the raw red/blue lines mix difficulty into the signal. Google
corrected for this by sampling 32 base-model responses per GPQA question and
using the fraction correct as that question's difficulty floor (§6 step 5, the
red line in their Fig 3). This module reproduces that: sample N single-shot
responses from the *base* model, score each against gold, and return
{goal_id: accuracy}. That dict feeds blue_minus_red_spread in concordance.py.

The reference must use the SAME model tier the co-scientist's Generation agent
uses (the fast model — agents/base.py defaults use_strong=False), so the blue−red
spread isolates the agentic scaffolding's contribution rather than a model-tier
gap. Override with use_strong=True only if you also move generation to strong.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from bench.baseline import SINGLE_SHOT_PROMPT
from bench.datasets.gpqa import score_answer
from bench.goalset import BenchGoal

GenerateFn = Callable[[str], Awaitable[str]]

# A minimal, non-empty system prompt: the base-model reference is a plain
# single-shot answer (no agentic scaffolding), and some backends reject an
# empty system block, so keep it short and neutral.
_REFERENCE_SYSTEM = "You are a careful scientific reasoner."


async def _sample_texts(
    goal: BenchGoal, generate: GenerateFn, n: int, sem: asyncio.Semaphore,
) -> list[str]:
    """n single-shot samples for one question, bounded by `sem`, tolerating
    per-sample failures (a failed call drops out rather than sinking the read)."""
    prompt = SINGLE_SHOT_PROMPT.format(goal=goal.goal)

    async def one() -> str:
        async with sem:
            return await generate(prompt)

    results = await asyncio.gather(*(one() for _ in range(n)), return_exceptions=True)
    return [r for r in results if isinstance(r, str)]


async def reference_accuracy_for_goals(
    goals: list[BenchGoal], generate: GenerateFn, n_samples: int = 32,
    *, concurrency: int = 8,
) -> dict[str, float]:
    """{goal_id: fraction of base-model samples whose parsed answer matches gold}.

    Only MCQ goals (those carrying a gold_answer) get a reference; open-ended
    goals are skipped. Per-sample failures are tolerated (accuracy is over the
    samples that returned); a goal with zero usable samples is skipped and
    reported. `concurrency` caps simultaneous in-flight calls so a large
    question set doesn't burst past the backend's rate limit. n_samples <= 0
    disables the feature (empty dict, no calls), preserving the default cost
    profile. Coverage is printed so a partial reference is never silent.
    """
    refs: dict[str, float] = {}
    if n_samples <= 0:
        return refs
    sem = asyncio.Semaphore(max(1, concurrency))
    mcq = [g for g in goals if g.gold_answer]
    skipped: list[str] = []
    for goal in mcq:
        texts = await _sample_texts(goal, generate, n_samples, sem)
        if not texts:
            skipped.append(goal.id)
            continue
        correct = sum(1 for t in texts if score_answer(t, goal.gold_answer))
        refs[goal.id] = correct / len(texts)
    if mcq:
        cov = 100 * len(refs) / len(mcq)
        print(f"  reference coverage: {len(refs)}/{len(mcq)} MCQ goals ({cov:.0f}%)"
              + (f" — skipped (no usable samples): "
                 f"{', '.join(skipped[:5])}{'…' if len(skipped) > 5 else ''}"
                 if skipped else ""))
    return refs


def backend_generate(backend, *, use_strong: bool = False) -> GenerateFn:
    """Adapt an LLMBackend.call into the single-prompt generate fn the sampler wants.

    Defaults to the fast model to match the Generation agent's tier (see module
    docstring): the reference is the same base model the system generates with,
    sampled single-shot, so blue−red measures the scaffolding, not a tier gap.
    """
    async def generate(prompt: str) -> str:
        return await backend.call(_REFERENCE_SYSTEM, prompt, use_strong=use_strong)
    return generate
