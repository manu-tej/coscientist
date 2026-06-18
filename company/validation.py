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

from dataclasses import dataclass, field
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


@dataclass
class ValidationReport:
    n_pairs: int
    n_scored: int
    n_gold: int
    auroc: float                    # gold vs decoy separation by score
    mean_gold: float
    mean_decoy: float
    negative_control_max: float     # highest-scoring decoy (should stay modest)
    recall_at_1: float              # per-disease: true drug is the top-ranked
    recall_at_3: float
    concordance: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)   # (drug, disease, score, is_gold)


def all_drugs() -> list[str]:
    return sorted({d for ds in GOLD.values() for d in ds})


def gold_pairs() -> list[tuple[str, str, bool]]:
    """Every (drug, disease) pair with its gold/decoy label (full cross-product)."""
    drugs = all_drugs()
    return [(drug, disease, drug in trues)
            for disease, trues in GOLD.items() for drug in drugs]


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


def validate(*, scorer: Optional[Scorer] = None, fetch=None) -> ValidationReport:
    """Score every gold/decoy pair and report separation, recall, and concordance.

    `scorer` overrides the signal entirely (tests pass a deterministic one);
    otherwise the live literature co-occurrence method is used (with `fetch`
    injectable for its network call). Pairs the scorer abstains on are excluded.
    """
    score_fn = scorer or _literature_scorer(fetch)
    rows = [(drug, disease, score_fn(drug, disease), is_gold)
            for drug, disease, is_gold in gold_pairs()]
    scored = [(d, dz, s, g) for (d, dz, s, g) in rows if s is not None]

    labels = [g for _, _, _, g in scored]
    scores = [s for _, _, s, _ in scored]
    gold_scores = [s for s, g in zip(scores, labels) if g]
    decoy_scores = [s for s, g in zip(scores, labels) if not g]

    concord = concordance_stats(
        [ScoredHypothesis(elo=1000.0 + 1000.0 * s, correct=g, question_id=dz)
         for (_, dz, s, g) in scored],
        bin_width=100, min_support=2,
    )

    return ValidationReport(
        n_pairs=len(rows), n_scored=len(scored), n_gold=sum(labels),
        auroc=_auroc(labels, scores),
        mean_gold=round(sum(gold_scores) / len(gold_scores), 4) if gold_scores else float("nan"),
        mean_decoy=round(sum(decoy_scores) / len(decoy_scores), 4) if decoy_scores else float("nan"),
        negative_control_max=round(max(decoy_scores), 4) if decoy_scores else float("nan"),
        recall_at_1=_recall_at_k(scored, 1),
        recall_at_3=_recall_at_k(scored, 3),
        concordance=concord,
        rows=rows,
    )
