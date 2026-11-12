from __future__ import annotations

from typing import Optional


def deterministic_ranking_tiebreaker(hypothesis: object) -> tuple[str, str, str]:
    """Return the content-first tie order shared by every ranking consumer."""
    return (
        str(getattr(hypothesis, "summary", "")).casefold(),
        str(getattr(hypothesis, "text", "")).casefold(),
        str(getattr(hypothesis, "id", "")),
    )


def native_ranking_key(
    hypothesis: object,
    score: Optional[float],
) -> tuple[bool, float, str, str, str]:
    """Return a best-first key, placing missing native scores last."""
    return (
        score is None,
        -float(score) if score is not None else 0.0,
        *deterministic_ranking_tiebreaker(hypothesis),
    )
