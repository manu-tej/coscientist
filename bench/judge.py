from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from typing import Optional, Protocol

from bench.errors import BenchError
from core.models import ABSOLUTE_SCORE_AXES

RUBRIC_AXES = list(ABSOLUTE_SCORE_AXES)
ABSOLUTE_RUBRIC_VERSION = "absolute-rubric-v1"
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
    Tolerates code fences and surrounding prose by grabbing the first {...} block.
    Raises BenchError on undecodable JSON or a missing/non-integer axis score."""
    blob = _extract_json_object(text)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise BenchError(f"Judge response is not valid JSON: {exc}") from exc
    scores: dict[str, int] = {}
    for axis in RUBRIC_AXES:
        node = data.get(axis, {})
        raw = node.get("score") if isinstance(node, dict) else node
        if raw is None:
            raise BenchError(f"Judge response missing score for axis {axis!r}")
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise BenchError(f"Non-integer score for axis {axis!r}: {raw!r}")
        score = raw
        if score < 1 or score > 5:
            raise BenchError(
                f"Score for axis {axis!r} must be within [1, 5], got {score!r}"
            )
        scores[axis] = score
    return scores


def _extract_json_object(text: str) -> str:
    """Return the JSON object string from a judge response. Prefers the first
    ```json ... ``` fenced block; else the first balanced {...} object. Raises
    BenchError if neither is found."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        obj = _first_balanced_object(fence.group(1))
        if obj is not None:
            return obj
    obj = _first_balanced_object(text)
    if obj is not None:
        return obj
    raise BenchError(f"No JSON object found in judge response: {text[:120]!r}")


def _first_balanced_object(text: str) -> Optional[str]:
    """Scan for the first brace-balanced top-level object (handles nesting),
    so we don't over-capture across multiple objects like a greedy regex would."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def weighted_total(scores: dict[str, int], weights: Optional[dict[str, float]] = None) -> float:
    w = normalize_rubric_weights(weights)
    num = sum(scores[a] * w[a] for a in RUBRIC_AXES)
    den = sum(w[a] for a in RUBRIC_AXES)
    return num / den


def normalize_rubric_weights(
    weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    raw = _DEFAULT_WEIGHTS if weights is None else weights
    if set(raw) != set(RUBRIC_AXES):
        raise BenchError(
            "rubric weights must contain exactly: " + ", ".join(RUBRIC_AXES)
        )
    normalized: dict[str, float] = {}
    for axis in RUBRIC_AXES:
        value = raw[axis]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise BenchError(
                f"rubric weight for {axis!r} must be finite and nonnegative"
            )
        normalized[axis] = float(value)
    if sum(normalized.values()) <= 0:
        raise BenchError("at least one rubric weight must be positive")
    return normalized


def absolute_scorer_identity(
    *,
    provider: str,
    model: str,
    weights: Optional[dict[str, float]] = None,
    panel_size: int = 1,
    decoding_tier: str = "strong",
) -> dict[str, object]:
    """Return the canonical, secret-free identity of an absolute judge."""
    if not provider.strip() or not model.strip():
        raise BenchError("absolute judge provider and model must be nonempty")
    if panel_size < 1:
        raise BenchError("absolute judge panel_size must be >= 1")
    if not decoding_tier.strip():
        raise BenchError("absolute judge decoding_tier must be nonempty")
    descriptor: dict[str, object] = {
        "provider": provider.strip(),
        "model": model.strip(),
        "rubric_version": ABSOLUTE_RUBRIC_VERSION,
        "rubric_sha256": hashlib.sha256(
            _JUDGE_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "axes": list(RUBRIC_AXES),
        "weights": normalize_rubric_weights(weights),
        "panel_size": panel_size,
        "decoding_tier": decoding_tier.strip(),
    }
    canonical = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return {
        **descriptor,
        "scorer_id": "absolute-" + hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }


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
        self.weights = normalize_rubric_weights(weights)

    async def score(self, hypothesis: str, field: str = "computational biology") -> dict:
        prompt = build_judge_prompt(hypothesis, field=field)
        raw = await self.backend.score_text("You are a rigorous scientific evaluator.", prompt)
        scores = parse_rubric_json(raw)
        return {"scores": scores, "total": weighted_total(scores, self.weights)}


def panel_median(panel: list[dict[str, int]]) -> dict[str, int]:
    """Aggregate a panel of per-axis score dicts by per-axis median (rounded to int)."""
    out: dict[str, int] = {}
    for axis in RUBRIC_AXES:
        vals = [p[axis] for p in panel if axis in p]
        out[axis] = int(round(statistics.median(vals))) if vals else 0
    return out


def pairwise_consistent_winner(
    winner_order1: str, winner_order2: str,
    order1: tuple[str, str], order2: tuple[str, str],
) -> Optional[str]:
    """A pairwise win counts only if consistent across position orders (§7).
    Returns the consistent winner id, or None (tie) on disagreement.
    Guards that order2 is genuinely a position swap of the same pair."""
    if set(order1) != set(order2):
        raise BenchError(
            f"position-swap requires the same pair in both orders, got {order1} and {order2}"
        )
    return winner_order1 if winner_order1 == winner_order2 else None


def krippendorff_alpha_per_axis(
    ratings_by_axis: dict[str, list[list[float]]]
) -> dict[str, float]:
    """Krippendorff's α (ordinal) per axis. Input: axis -> reliability matrix
    (rows = judges, cols = items)."""
    import krippendorff

    out: dict[str, float] = {}
    for axis, matrix in ratings_by_axis.items():
        out[axis] = float(
            krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")
        )
    return out


async def score_panel(
    judges: list["Judge"], hypothesis: str, field: str = "computational biology"
) -> dict:
    """Score one hypothesis with a panel of ≥3 judges; aggregate by median."""
    import asyncio
    results = await asyncio.gather(*(j.score(hypothesis, field=field) for j in judges))
    per_axis = [r["scores"] for r in results]
    med = panel_median(per_axis)
    return {"scores": med, "total": weighted_total(med),
            "panel": per_axis, "n_judges": len(judges)}
