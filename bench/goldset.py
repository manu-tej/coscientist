from __future__ import annotations

import re
import unicodedata
from typing import Optional, Protocol

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(s: str) -> list[str]:
    """Unicode-normalize, casefold, split into alphanumeric tokens.
    Hyphens/underscores/punctuation become token boundaries."""
    norm = unicodedata.normalize("NFKC", s).casefold()
    return _TOKEN_RE.findall(norm)


def _is_contiguous_subsequence(needle: list[str], haystack: list[str]) -> bool:
    if not needle:
        return False
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return True
    return False


def entity_in_text(entity: str, text: str) -> bool:
    """True if the entity's token sequence appears contiguously in text."""
    return _is_contiguous_subsequence(tokenize(entity), tokenize(text))


class _HasText(Protocol):
    text: str
    summary: str


def _searched_fields(h: _HasText) -> str:
    return f"{getattr(h, 'summary', '')} {getattr(h, 'text', '')}"


def score_recall(
    hypotheses: list[_HasText],
    gold_entities: list[str],
    k: Optional[int] = None,
) -> float:
    """recall = (gold entities found in any hypothesis) / (total gold entities).
    If k is given, only the top-k hypotheses by elo_rating are searched."""
    if not gold_entities:
        return 0.0
    pool = hypotheses
    if k is not None:
        pool = sorted(hypotheses, key=lambda h: getattr(h, "elo_rating", 0.0), reverse=True)[:k]
    # Join all searched hypotheses into one blob. Tradeoff: a multi-token entity
    # could in principle match across a hypothesis boundary; acceptable for recall.
    blob = " ".join(_searched_fields(h) for h in pool)
    hits = sum(1 for e in gold_entities if entity_in_text(e, blob))
    return hits / len(gold_entities)
