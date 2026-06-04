from __future__ import annotations

import math
import random as _random
from collections import defaultdict
from dataclasses import dataclass

from scipy.stats import spearmanr, kendalltau


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


def concordance_stats(
    rows: list[ScoredHypothesis], bin_width: int = 50, min_support: int = 5
) -> dict:
    """Spearman ρ, Kendall τ-b (bucket midpoint vs accuracy), and a response-level
    logistic regression correct ~ elo. Returns a flat dict of statistics."""
    acc = per_bucket_accuracy(rows, bin_width, min_support)
    floors = sorted(acc.keys())
    midpoints = [f + bin_width / 2 for f in floors]
    accuracies = [acc[f] for f in floors]

    if len(floors) >= 3:
        rho, rho_p = spearmanr(midpoints, accuracies)
        tau, tau_p = kendalltau(midpoints, accuracies)
    else:
        rho = rho_p = tau = tau_p = float("nan")

    logit_coef, logit_p = _logistic_correct_on_elo(rows)

    return {
        "n_rows": len(rows),
        "n_buckets": len(floors),
        "bucket_floors": floors,
        "bucket_accuracy": accuracies,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "logistic_coef": logit_coef,
        "logistic_p": logit_p,
        "top1_accuracy": top1_accuracy(rows),
    }


def _logistic_correct_on_elo(rows: list[ScoredHypothesis]) -> tuple[float, float]:
    """Fit correct ~ elo via statsmodels GLM (binomial). Returns (coef, p) for elo.
    Falls back to (nan, nan) if degenerate (all-correct / all-wrong / singular)."""
    import numpy as np
    import statsmodels.api as sm

    y = np.array([1.0 if r.correct else 0.0 for r in rows])
    if y.sum() == 0 or y.sum() == len(y) or len(rows) < 5:
        return float("nan"), float("nan")
    # Center & scale Elo for numerical stability; coef is per-scaled-unit but sign/p hold.
    elo = np.array([r.elo for r in rows])
    x = (elo - elo.mean()) / (elo.std() or 1.0)
    X = sm.add_constant(x)
    try:
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        return float(model.params[1]), float(model.pvalues[1])
    except Exception:
        return float("nan"), float("nan")


def reference_per_bucket(
    rows: list[ScoredHypothesis],
    reference_accuracy: dict[str, float],
    bin_width: int = 50,
    min_support: int = 5,
) -> dict[int, float]:
    """Difficulty-correction 'red line': for each Elo bucket, the mean base-model
    accuracy over the *questions* contributing hypotheses to that bucket."""
    buckets = bucket_by_elo(rows, bin_width)
    out: dict[int, float] = {}
    for floor, items in buckets.items():
        if len(items) < min_support:
            continue
        qids = [it.question_id for it in items]
        refs = [reference_accuracy[q] for q in qids if q in reference_accuracy]
        if refs:
            out[floor] = sum(refs) / len(refs)
    return out


def blue_minus_red_spread(
    rows: list[ScoredHypothesis],
    reference_accuracy: dict[str, float],
    bin_width: int = 50,
    min_support: int = 5,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Mean (blue − red) bucket spread with a cluster bootstrap that resamples
    whole questions (hypotheses within a question are correlated, §17.1)."""
    blue = per_bucket_accuracy(rows, bin_width, min_support)
    red = reference_per_bucket(rows, reference_accuracy, bin_width, min_support)
    common = sorted(set(blue) & set(red))
    if not common:
        return {"mean_spread": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_buckets": 0}

    point = sum(blue[f] - red[f] for f in common) / len(common)

    by_q: dict[str, list[ScoredHypothesis]] = defaultdict(list)
    for r in rows:
        by_q[r.question_id].append(r)
    qids = list(by_q.keys())
    rng = _random.Random(seed)

    boot_means: list[float] = []
    for _ in range(n_boot):
        sample_rows: list[ScoredHypothesis] = []
        for _ in range(len(qids)):
            q = rng.choice(qids)
            sample_rows.extend(by_q[q])
        b = per_bucket_accuracy(sample_rows, bin_width, min_support)
        r_ = reference_per_bucket(sample_rows, reference_accuracy, bin_width, min_support)
        c = sorted(set(b) & set(r_))
        if c:
            boot_means.append(sum(b[f] - r_[f] for f in c) / len(c))
    if not boot_means:
        return {"mean_spread": point, "ci_low": float("nan"),
                "ci_high": float("nan"), "n_buckets": len(common)}
    boot_means.sort()
    lo = boot_means[int(0.025 * len(boot_means))]
    hi = boot_means[int(0.975 * len(boot_means)) - 1]
    return {"mean_spread": point, "ci_low": lo, "ci_high": hi, "n_buckets": len(common)}
