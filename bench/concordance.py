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
    # The tournament this Elo came from. Elo is comparable only within one
    # tournament, so the conditional fit stratifies on this; it defaults to the
    # question because the benchmark runs each question once. question_id stays
    # the key for per-question reference accuracy (the difficulty red line).
    stratum: str | None = None


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
    """Mean correctness of the top-Elo hypothesis (ties averaged), taken within
    each tournament, averaged over a question's tournaments, then over
    questions. Elo is comparable only within one tournament, so the top is
    never chosen across tournaments of the same question."""
    by_question: dict[str, dict[str, list[ScoredHypothesis]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_question[r.question_id][r.stratum or r.question_id].append(r)
    if not by_question:
        return 0.0
    per_question = []
    for tournaments in by_question.values():
        per_tournament = []
        for tournament_rows in tournaments.values():
            top_elo = max(row.elo for row in tournament_rows)
            tied = [row for row in tournament_rows if row.elo == top_elo]
            per_tournament.append(sum(row.correct for row in tied) / len(tied))
        per_question.append(sum(per_tournament) / len(per_tournament))
    return sum(per_question) / len(per_question)


def concordance_stats(
    rows: list[ScoredHypothesis], bin_width: int = 50, min_support: int = 5
) -> dict:
    """Flat dict of concordance statistics. Headline: a conditional logistic
    regression correct ~ elo stratified by tournament (see
    ``_conditional_logistic_correct_on_elo``). Descriptive: Spearman ρ and
    Kendall τ-b over bucket midpoints vs per-bucket accuracy, and top-1 accuracy."""
    acc = per_bucket_accuracy(rows, bin_width, min_support)
    floors = sorted(acc.keys())
    midpoints = [f + bin_width / 2 for f in floors]
    accuracies = [acc[f] for f in floors]

    if len(floors) >= 3:
        rho, rho_p = spearmanr(midpoints, accuracies)
        tau, tau_p = kendalltau(midpoints, accuracies)
    else:
        rho = rho_p = tau = tau_p = float("nan")

    return {
        "n_rows": len(rows),
        "n_buckets": len(floors),
        "bucket_floors": floors,
        "bucket_accuracy": accuracies,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        **_conditional_logistic_correct_on_elo(rows),
        "top1_accuracy": top1_accuracy(rows),
    }


def _conditional_logistic_correct_on_elo(rows: list[ScoredHypothesis]) -> dict:
    """Conditional logistic regression correct ~ elo, stratified by tournament.

    Every question is its own tournament: all hypotheses start at the same Elo
    and drift only against siblings from the same tournament, so raw Elo is not
    on a common scale across tournaments and a pooled fit would mix question
    difficulty into the estimate. Conditioning on the tournament (``stratum``,
    defaulting to ``question_id``) estimates only the within-tournament effect,
    which is the concordance claim. ``logistic_n_questions`` counts strata.

    Elo is centered within each question and scaled by the pooled SD of the
    centered values, so ``logistic_coef`` is log-odds per SD of within-tournament
    Elo. ``logistic_p`` is a likelihood-ratio test against the zero-slope model.
    Questions with no outcome variance (all correct or all wrong) or no Elo
    variance (a tournament that never ran, every hypothesis still tied) carry no
    information under this model and are excluded; ``logistic_n_rows`` and
    ``logistic_n_questions`` report the informative subset actually fitted.
    Fewer than two informative questions or five informative rows gives NaN.
    Under (quasi-)complete within-question separation the MLE diverges and the
    optimizer stops at an arbitrarily large slope; the estimate and LRT p are
    still reported and ``logistic_separated`` is True so the magnitude is not
    read as meaningful."""
    import warnings

    import numpy as np
    from scipy.stats import chi2

    by_question: dict[str, list[ScoredHypothesis]] = defaultdict(list)
    for r in rows:
        by_question[r.stratum or r.question_id].append(r)
    informative = {
        q: items for q, items in by_question.items()
        if any(i.correct for i in items) and not all(i.correct for i in items)
        and len({i.elo for i in items}) > 1
    }
    n_rows = sum(len(items) for items in informative.values())
    out = {
        "logistic_method": "conditional_by_tournament",
        "logistic_coef": float("nan"),
        "logistic_p": float("nan"),
        "logistic_n_rows": n_rows,
        "logistic_n_questions": len(informative),
        "logistic_separated": False,
    }
    if len(informative) < 2 or n_rows < 5:
        return out

    y: list[float] = []
    x: list[float] = []
    groups: list[int] = []
    for gi, items in enumerate(informative.values()):
        mean = sum(i.elo for i in items) / len(items)
        for i in items:
            y.append(1.0 if i.correct else 0.0)
            x.append(i.elo - mean)
            groups.append(gi)
    x_arr = np.array(x)
    x_arr = x_arr / (x_arr.std() or 1.0)

    from statsmodels.discrete.conditional_models import ConditionalLogit

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # drop notices / convergence are reported below
        try:
            model = ConditionalLogit(np.array(y), x_arr[:, None], groups=np.array(groups))
            fit = model.fit(disp=0, maxiter=200)
            coef = float(fit.params[0])
            llf_alt = float(fit.llf)
            llf_null = float(model.loglike(np.zeros(1)))
        except Exception:
            return out

    lr_stat = max(0.0, 2.0 * (llf_alt - llf_null))
    out["logistic_coef"] = coef
    out["logistic_p"] = float(chi2.sf(lr_stat, df=1))
    out["logistic_separated"] = _separated_within_every_stratum(informative.values())
    return out


def _separated_within_every_stratum(strata) -> bool:
    """True when one direction puts every correct row's Elo on or beyond every
    incorrect row's Elo in EVERY stratum. Then the conditional likelihood keeps
    rising with the slope and the MLE is infinite (quasi-complete separation)."""
    positive = negative = True
    for items in strata:
        correct = [i.elo for i in items if i.correct]
        wrong = [i.elo for i in items if not i.correct]
        positive = positive and min(correct) >= max(wrong)
        negative = negative and max(correct) <= min(wrong)
        if not (positive or negative):
            return False
    return True


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
        # Hypothesis-count-weighted (NOT unique-question): a question contributing
        # k hypotheses to this bucket counts k times. This matches per_bucket_accuracy
        # (blue), which is also per-hypothesis, so blue-red stays apples-to-apples.
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
    n = len(boot_means)
    lo = boot_means[min(int(0.025 * n), n - 1)]
    hi = boot_means[min(int(0.975 * n), n - 1)]
    return {"mean_spread": point, "ci_low": lo, "ci_high": hi, "n_buckets": len(common)}
