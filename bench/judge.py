from __future__ import annotations

import json
import re
from typing import Optional, Protocol

from bench.errors import BenchError

RUBRIC_AXES = ["novelty", "feasibility", "correctness", "impact"]
_DEFAULT_WEIGHTS = {a: 1.0 for a in RUBRIC_AXES}

_JUDGE_TEMPLATE = """You are an expert evaluator in {field}. Score the hypothesis
below on each axis from 1 (poor) to 5 (excellent). Be calibrated and critical.

Axes:
- novelty: is the idea new and non-obvious?
- feasibility: can it realistically be tested with current methods?
- correctness: is the underlying reasoning scientifically sound?
- impact: would confirming it meaningfully advance the field?

Hypothesis:
{hypothesis}

Respond with ONLY a JSON object of the form:
{{"novelty": {{"score": N, "justification": "..."}}, "feasibility": {{...}},
  "correctness": {{...}}, "impact": {{...}}}}
"""


class JudgeBackend(Protocol):
    async def score_text(self, system: str, user: str) -> str: ...


def build_judge_prompt(hypothesis: str, field: str = "computational biology") -> str:
    return _JUDGE_TEMPLATE.format(field=field, hypothesis=hypothesis)


def parse_rubric_json(text: str) -> dict[str, int]:
    """Extract per-axis integer scores from the judge's text response.
    Tolerates code fences and surrounding prose by grabbing the first {...} block."""
    blob = _extract_json_object(text)
    data = json.loads(blob)
    scores: dict[str, int] = {}
    for axis in RUBRIC_AXES:
        node = data.get(axis, {})
        if isinstance(node, dict):
            scores[axis] = int(node.get("score"))
        else:
            scores[axis] = int(node)
    return scores


def _extract_json_object(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return brace.group(0)
    raise BenchError(f"No JSON object found in judge response: {text[:120]!r}")


def weighted_total(scores: dict[str, int], weights: Optional[dict[str, float]] = None) -> float:
    w = weights or _DEFAULT_WEIGHTS
    num = sum(scores[a] * w[a] for a in RUBRIC_AXES)
    den = sum(w[a] for a in RUBRIC_AXES)
    return num / den


class Judge:
    """A bias-controlled LLM judge, decoupled from the system-under-test."""

    def __init__(
        self,
        backend: JudgeBackend,
        judge_model: str,
        generator_model: str,
        weights: Optional[dict[str, float]] = None,
    ):
        if judge_model == generator_model:
            raise BenchError(
                f"Judge model must differ from generator model (both are {judge_model!r}). "
                "Self-judging defeats bias control (§7)."
            )
        self.backend = backend
        self.judge_model = judge_model
        self.generator_model = generator_model
        self.weights = weights or _DEFAULT_WEIGHTS

    async def score(self, hypothesis: str, field: str = "computational biology") -> dict:
        prompt = build_judge_prompt(hypothesis, field=field)
        raw = await self.backend.score_text("You are a rigorous scientific evaluator.", prompt)
        scores = parse_rubric_json(raw)
        return {"scores": scores, "total": weighted_total(scores, self.weights)}
