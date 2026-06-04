from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ScoredHypothesis:
    elo: float
    correct: bool
    question_id: str


def _bin_floor(elo: float, bin_width: int) -> int:
    return int(math.floor(elo / bin_width) * bin_width)


def bucket_by_elo(
    rows: list[ScoredHypothesis], bin_width: int = 50
) -> dict[int, list[ScoredHypothesis]]:
    """Group scored hypotheses into bin_width-point Elo buckets keyed by bin floor."""
    buckets: dict[int, list[ScoredHypothesis]] = defaultdict(list)
    for r in rows:
        buckets[_bin_floor(r.elo, bin_width)].append(r)
    return dict(buckets)


def per_bucket_accuracy(
    rows: list[ScoredHypothesis], bin_width: int = 50, min_support: int = 5
) -> dict[int, float]:
    """Fraction correct per Elo bucket, dropping buckets below min_support."""
    buckets = bucket_by_elo(rows, bin_width)
    out: dict[int, float] = {}
    for floor, items in buckets.items():
        if len(items) < min_support:
            continue
        out[floor] = sum(1 for x in items if x.correct) / len(items)
    return out


def top1_accuracy(rows: list[ScoredHypothesis]) -> float:
    """Accuracy of the single highest-Elo hypothesis per question."""
    best: dict[str, ScoredHypothesis] = {}
    for r in rows:
        cur = best.get(r.question_id)
        if cur is None or r.elo > cur.elo:
            best[r.question_id] = r
    if not best:
        return 0.0
    return sum(1 for r in best.values() if r.correct) / len(best)
