"""The company's report card — validation against known repurposings (spec §11).

Is the system's confidence *real*? We hold out famous, established drug→indication
repurposings as ground truth and ask: does the live evidence signal rank the true
repurposings above decoy (drug, wrong-disease) pairs?

Every gold drug is also a *decoy* for every disease it does NOT treat — so the
negative controls are built in (minoxidil is true for hair-loss, a decoy for
myeloma). The scorer is injectable: the live path uses the literature
co-occurrence method (real, network), while tests pass a deterministic scorer to
exercise the metrics offline. The concordance stat reuses bench/concordance.py
directly, as the spec intends ("reuses bench concordance").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Callable, Optional

from bench.concordance import ScoredHypothesis, concordance_stats

# Curated gold set: disease → established repurposed (or signature) drugs. Small,
# famous, and contamination-resistant as named test cases (spec §11).
GOLD: dict[str, list[str]] = {
    "pulmonary arterial hypertension": ["sildenafil", "tadalafil"],
    "multiple myeloma": ["thalidomide", "lenalidomide"],
    "androgenetic alopecia": ["minoxidil", "finasteride"],
    "breast cancer": ["raloxifene", "tamoxifen"],
    "erectile dysfunction": ["sildenafil"],
    "type 2 diabetes": ["metformin"],
}

# A score in [0, 1] for a (drug, disease) pair, or None to abstain (unavailable).
Scorer = Callable[[str, str], Optional[float]]


class ValidationCoverageError(RuntimeError):
    """A scorer abstained, so a complete labeled matrix was not observed."""

    def __init__(
        self,
        expected: int,
        scored: int,
        missing_pairs: list[tuple[str, str]],
    ):
        super().__init__(
            f"validation coverage incomplete: scored {scored}/{expected} pairs"
        )
        self.expected = expected
        self.scored = scored
        self.missing_pairs = list(missing_pairs)


class ValidationDataError(ValueError):
    """A scorer returned a value outside the declared [0, 1] contract."""


@dataclass
class ValidationReport:
    n_pairs: int
    n_scored: int
    n_gold: int                       # compatibility: scored gold rows
    auroc: float                    # gold vs decoy separation by score
    mean_gold: float
    mean_decoy: float
    negative_control_max: float     # highest-scoring decoy (should stay modest)
    recall_at_1: float              # per-disease: true drug is the top-ranked
    recall_at_3: float
    concordance: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)   # (drug, disease, score, is_gold)
    # Appended defaults preserve the original positional construction contract.
    n_gold_expected: int = 0
    n_gold_scored: int = 0
    n_missing: int = 0
    coverage: float = 0.0
    complete: bool = False
    mode: str = ""
    source: str = ""


def all_drugs() -> list[str]:
    _validate_gold_definition()
    return sorted({d for ds in GOLD.values() for d in ds})


def _validate_gold_definition() -> None:
    """Fail before any scorer call when the curated labeled matrix is malformed."""
    if not isinstance(GOLD, dict) or not GOLD:
        raise ValidationDataError("curated GOLD set must contain at least one disease")
    canonical_drugs: dict[str, str] = {}
    for disease, drugs in GOLD.items():
        if not isinstance(disease, str) or not disease.strip():
            raise ValidationDataError(
                "curated GOLD disease names must be non-blank strings"
            )
        if not isinstance(drugs, list) or not drugs:
            raise ValidationDataError(
                f"curated GOLD disease {disease!r} must have at least one gold drug"
            )
        normalized: list[str] = []
        for drug in drugs:
            if not isinstance(drug, str) or not drug.strip():
                raise ValidationDataError(
                    f"curated GOLD disease {disease!r} has an invalid drug name"
                )
            key = drug.strip().casefold()
            normalized.append(key)
            previous = canonical_drugs.setdefault(key, drug)
            if previous != drug:
                raise ValidationDataError(
                    f"curated GOLD drug {drug!r} has inconsistent spelling/case"
                )
        if len(set(normalized)) != len(normalized):
            raise ValidationDataError(
                f"curated GOLD disease {disease!r} contains duplicate gold drugs"
            )


def gold_pairs() -> list[tuple[str, str, bool]]:
    """Every (drug, disease) pair with its gold/decoy label (full cross-product)."""
    _validate_gold_definition()
    drugs = all_drugs()
    pairs = [
        (drug, disease, drug in trues)
        for disease, trues in GOLD.items()
        for drug in drugs
    ]
    labels = {is_gold for *_, is_gold in pairs}
    if labels != {False, True}:
        raise ValidationDataError(
            "curated GOLD cross-product must contain both gold and decoy labels"
        )
    return pairs


def _auroc(labels: list[bool], scores: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score([1 if l else 0 for l in labels], scores))


def _recall_at_k(rows: list[tuple[str, str, float, bool]], k: int) -> float:
    """Per disease, rank its scored drugs; fraction of diseases whose true drug(s)
    appear in the top-k. Averaged over diseases that have at least one gold + score."""
    by_disease: dict[str, list[tuple[str, float, bool]]] = {}
    for drug, disease, score, is_gold in rows:
        by_disease.setdefault(disease, []).append((drug, score, is_gold))
    hits, n = 0, 0
    for disease, items in by_disease.items():
        if not any(g for _, _, g in items):
            continue
        n += 1
        ranked = sorted(items, key=lambda t: t[1], reverse=True)[:k]
        if any(g for _, _, g in ranked):
            hits += 1
    return hits / n if n else float("nan")


def _literature_scorer(fetch=None) -> Scorer:
    from company import literature
    return lambda drug, disease: literature.literature_score(drug, disease, fetch=fetch)


def validate(
    *,
    scorer: Optional[Scorer] = None,
    fetch=None,
    require_complete: bool = True,
) -> ValidationReport:
    """Score every gold/decoy pair and report separation, recall, and concordance.

    `scorer` overrides the signal entirely (tests pass a deterministic one);
    otherwise the live literature co-occurrence method is used (with `fetch`
    injectable for its network call). Strict mode is the default: every labeled
    pair must produce a finite [0, 1] score. Diagnostic partial reports require
    the explicit ``require_complete=False`` opt-in and are marked incomplete.
    """
    score_fn = (
        scorer
        if scorer is not None
        else _literature_scorer(fetch)
    )
    if scorer is not None:
        mode = "injected_scorer"
        source = "caller-provided scorer"
    elif fetch is not None:
        mode = "injected_fetch"
        source = "caller-provided PubMed count fetch"
    else:
        mode = "live_pubmed"
        source = "NCBI PubMed co-occurrence"
    rows: list[tuple[str, str, float | None, bool]] = []
    missing_pairs: list[tuple[str, str]] = []
    for drug, disease, is_gold in gold_pairs():
        raw_score = score_fn(drug, disease)
        if raw_score is None:
            missing_pairs.append((drug, disease))
            rows.append((drug, disease, None, is_gold))
            continue
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, Real)
            or not math.isfinite(float(raw_score))
            or not 0 <= float(raw_score) <= 1
        ):
            raise ValidationDataError(
                f"invalid score for ({drug!r}, {disease!r}): "
                "expected a finite real number in [0, 1]"
            )
        rows.append((drug, disease, float(raw_score), is_gold))

    scored = [(d, dz, s, g) for (d, dz, s, g) in rows if s is not None]
    if require_complete and missing_pairs:
        raise ValidationCoverageError(
            len(rows),
            len(scored),
            missing_pairs,
        )

    labels = [g for _, _, _, g in scored]
    scores = [s for _, _, s, _ in scored]
    gold_scores = [s for s, g in zip(scores, labels) if g]
    decoy_scores = [s for s, g in zip(scores, labels) if not g]

    concord = concordance_stats(
        [ScoredHypothesis(elo=1000.0 + 1000.0 * s, correct=g, question_id=dz)
         for (_, dz, s, g) in scored],
        bin_width=100, min_support=2,
    )

    n_gold_expected = sum(1 for *_, is_gold in rows if is_gold)
    n_gold_scored = sum(labels)
    return ValidationReport(
        n_pairs=len(rows),
        n_scored=len(scored),
        n_gold=n_gold_scored,
        n_gold_expected=n_gold_expected,
        n_gold_scored=n_gold_scored,
        n_missing=len(missing_pairs),
        coverage=(len(scored) / len(rows)) if rows else 0.0,
        complete=not missing_pairs,
        mode=mode,
        source=source,
        auroc=_auroc(labels, scores),
        mean_gold=round(sum(gold_scores) / len(gold_scores), 4) if gold_scores else float("nan"),
        mean_decoy=round(sum(decoy_scores) / len(decoy_scores), 4) if decoy_scores else float("nan"),
        negative_control_max=round(max(decoy_scores), 4) if decoy_scores else float("nan"),
        recall_at_1=_recall_at_k(scored, 1),
        recall_at_3=_recall_at_k(scored, 3),
        concordance=concord,
        rows=rows,
    )
