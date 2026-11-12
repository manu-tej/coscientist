import pytest
from bench.concordance import (
    ScoredHypothesis, bucket_by_elo, per_bucket_accuracy, top1_accuracy,
)
from bench.concordance import (
    concordance_stats, reference_per_bucket, blue_minus_red_spread,
)


def _sh(elo, correct, qid="q1"):
    return ScoredHypothesis(elo=elo, correct=correct, question_id=qid)


def test_bucket_by_elo_50pt_bins():
    rows = [_sh(1010, True), _sh(1040, False), _sh(1055, True), _sh(1099, True)]
    buckets = bucket_by_elo(rows, bin_width=50)
    assert set(buckets.keys()) == {1000, 1050}     # bin floor keys
    assert len(buckets[1000]) == 2
    assert len(buckets[1050]) == 2


def test_per_bucket_accuracy_min_support():
    rows = [_sh(1010, True), _sh(1020, True), _sh(1030, False),
            _sh(1060, True)]   # bucket 1050 has only 1 → dropped at min_support=2
    acc = per_bucket_accuracy(rows, bin_width=50, min_support=2)
    assert acc[1000] == 2 / 3
    assert 1050 not in acc


def test_top1_accuracy_per_question():
    rows = [_sh(1300, True, "qA"), _sh(1100, False, "qA"),
            _sh(1250, False, "qB"), _sh(1200, True, "qB")]
    assert top1_accuracy(rows) == 0.5


def test_top1_accuracy_fractionally_scores_equal_elo_ties():
    rows = [
        _sh(1300, True, "qA"),
        _sh(1300, False, "qA"),
        _sh(1200, False, "qA"),
        _sh(1250, True, "qB"),
        _sh(1250, True, "qB"),
    ]

    expected = 0.75
    assert top1_accuracy(rows) == expected
    assert top1_accuracy(list(reversed(rows))) == expected


def test_top1_accuracy_is_taken_per_tournament_then_averaged_per_question():
    # Two tournaments on one question: run-A's top hypothesis is right, run-B's
    # is wrong but sits in a higher Elo band. Elo is not comparable across
    # tournaments, so top-1 must be (1 + 0) / 2 for the question, not 0.
    rows = [ScoredHypothesis(1240, True, "q", stratum="run-A"),
            ScoredHypothesis(1200, False, "q", stratum="run-A"),
            ScoredHypothesis(1300, False, "q", stratum="run-B"),
            ScoredHypothesis(1260, True, "q", stratum="run-B"),
            ScoredHypothesis(1250, True, "other")]
    assert top1_accuracy(rows) == 0.75                      # mean(0.5, 1.0)
    shifted = [ScoredHypothesis(r.elo - (100 if r.stratum == "run-B" else 0), r.correct,
                                r.question_id, stratum=r.stratum) for r in rows]
    assert top1_accuracy(shifted) == 0.75                   # band shift is irrelevant


def test_concordance_stats_monotonic_signal():
    # Construct a clean monotone elo→accuracy relationship across 14 buckets.
    rows = []
    for b in range(14):
        floor = 1000 + b * 50
        p_correct = b / 13.0            # rises 0 → 1
        for i in range(25):
            correct = i < round(p_correct * 25)
            rows.append(ScoredHypothesis(elo=floor + 25, correct=correct,
                                         question_id=f"q{i}"))
    stats = concordance_stats(rows, bin_width=50, min_support=5)
    assert stats["spearman_rho"] >= 0.7
    assert stats["spearman_p"] < 0.05
    assert stats["kendall_tau"] > 0
    # Every question here is perfectly separated (a step in b), so the
    # conditional MLE diverges: sign holds, magnitude does not.
    assert stats["logistic_coef"] > 0
    assert stats["logistic_separated"] is True
    assert stats["n_buckets"] >= 10


def test_reference_per_bucket_difficulty_correction():
    rows = [ScoredHypothesis(1010, True, "qA"), ScoredHypothesis(1060, True, "qA"),
            ScoredHypothesis(1010, False, "qB")]
    ref = {"qA": 0.5, "qB": 0.25}      # per-question base-model accuracy
    red = reference_per_bucket(rows, ref, bin_width=50, min_support=1)
    # bucket 1000 has qA + qB → mean(0.5, 0.25)=0.375 ; bucket 1050 has qA → 0.5
    assert abs(red[1000] - 0.375) < 1e-9
    assert abs(red[1050] - 0.5) < 1e-9


def test_blue_minus_red_spread_ci():
    rows, ref = [], {}
    for q in range(10):
        ref[f"q{q}"] = 0.3
        rows.append(ScoredHypothesis(1300, True, f"q{q}"))   # system beats 0.3
    res = blue_minus_red_spread(rows, ref, bin_width=50, min_support=1, n_boot=200, seed=1)
    assert res["mean_spread"] > 0
    assert res["ci_low"] <= res["mean_spread"] <= res["ci_high"]


# --- headline estimator: conditional logistic stratified by question -------------
#
# Every question is its own tournament (all hypotheses start at 1200 and drift only
# against siblings), so the headline must estimate the WITHIN-question effect of Elo
# on correctness and nothing else.

def _question(qid, center, pattern, step=10):
    """Rows for one question: Elo spread symmetrically around `center`, correctness
    given by `pattern` in ascending-Elo order."""
    n = len(pattern)
    offsets = [(i - (n - 1) / 2) * step for i in range(n)]
    return [ScoredHypothesis(elo=center + off, correct=bool(c), question_id=qid)
            for off, c in zip(offsets, pattern)]


def test_conditional_logistic_detects_within_question_signal():
    # Within every question, higher Elo is (noisily) more often correct.
    # One flip on each side of the boundary keeps every stratum non-separable.
    pattern = [0, 0, 0, 1, 0, 1, 1, 1]
    rows = []
    for q in range(6):
        rows += _question(f"q{q}", 1200, pattern)
    stats = concordance_stats(rows, bin_width=50, min_support=5)
    assert stats["logistic_method"] == "conditional_by_tournament"
    assert stats["logistic_coef"] > 0
    assert stats["logistic_p"] < 0.05
    assert stats["logistic_n_questions"] == 6
    assert stats["logistic_n_rows"] == 48
    assert stats["logistic_separated"] is False


def test_conditional_logistic_is_blind_to_between_question_difficulty():
    # Confound: easier questions also sit at higher Elo bands, so a POOLED view
    # (the bucketed Spearman) sees a positive trend. Within each question the
    # correct answers are the LOWER-Elo ones, so the within-question effect is
    # negative. The headline must report the within-question effect.
    rows = []
    for q in range(6):
        n_correct = q + 2                       # 2..7 of 8 correct: accuracy rises with q
        pattern = [1] * n_correct + [0] * (8 - n_correct)   # correct = lowest Elo
        pattern[n_correct - 1], pattern[n_correct] = 0, 1   # one boundary swap: non-separable
        rows += _question(f"q{q}", 1100 + 60 * q, pattern)   # Elo band rises with q
    stats = concordance_stats(rows, bin_width=100, min_support=3)
    assert stats["n_buckets"] >= 3
    assert stats["spearman_rho"] > 0            # the pooled, confounded view
    assert stats["logistic_coef"] < 0           # the within-question truth
    assert stats["logistic_p"] < 0.05


def test_conditional_logistic_drops_and_counts_uninformative_questions():
    import warnings
    pattern = [0, 0, 0, 1, 0, 1, 1, 1]
    rows = []
    for q in range(4):
        rows += _question(f"q{q}", 1200, pattern)
    rows += _question("all_wrong", 1200, [0] * 6)
    rows += _question("all_right", 1200, [1] * 5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # nothing may leak to the caller
        stats = concordance_stats(rows, bin_width=50, min_support=5)
    assert stats["n_rows"] == 43
    assert stats["logistic_n_rows"] == 32
    assert stats["logistic_n_questions"] == 4
    assert stats["logistic_coef"] > 0


def test_conditional_logistic_needs_two_informative_questions():
    import math
    rows = _question("only", 1200, [0, 0, 0, 1, 0, 1, 1, 1])
    stats = concordance_stats(rows, bin_width=50, min_support=5)
    assert math.isnan(stats["logistic_coef"])
    assert math.isnan(stats["logistic_p"])
    assert stats["logistic_n_questions"] == 1


def test_conditional_logistic_flags_within_question_separation():
    # Perfectly separable within every stratum: the MLE diverges and the
    # optimizer stops at some large slope. The sign and the LRT p are still
    # valid; the report flags separation so the magnitude is not over-read.
    import warnings
    rows = []
    for q in range(6):
        rows += _question(f"q{q}", 1200, [0, 0, 0, 0, 1, 1, 1, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats = concordance_stats(rows, bin_width=50, min_support=5)
    assert stats["logistic_coef"] > 0
    assert stats["logistic_p"] < 0.05
    assert stats["logistic_separated"] is True


# --- calibration of the headline number itself ---------------------------------

def _null_rows(n_questions=30, per_question=8, seed=0):
    """Correctness independent of Elo: a fixed 4-of-8 pattern, shuffled per
    question by a seeded RNG, so any within-question association is chance."""
    import random
    rng = random.Random(seed)
    rows = []
    for q in range(n_questions):
        pattern = [1, 1, 1, 1, 0, 0, 0, 0][:per_question]
        rng.shuffle(pattern)
        rows += _question(f"q{q}", 1200, pattern)
    return rows


def test_conditional_logistic_null_data_is_not_significant():
    from bench.report import _concordance_pass
    stats = concordance_stats(_null_rows(), bin_width=50, min_support=5)
    assert stats["logistic_n_questions"] == 30
    assert stats["logistic_p"] > 0.05
    assert _concordance_pass(stats) is False


def _brute_force_conditional_fit(rows):
    """Independent reference: exact conditional log-likelihood by enumerating,
    per question, every subset of the same size as the observed correct set;
    maximised over the slope by 1-D search; LRT against slope 0."""
    import itertools
    import math
    from collections import defaultdict
    import numpy as np
    from scipy.optimize import minimize_scalar
    from scipy.stats import chi2

    by_q = defaultdict(list)
    for r in rows:
        by_q[r.question_id].append(r)
    strata = []
    centered = []
    for items in by_q.values():
        if all(i.correct for i in items) or not any(i.correct for i in items):
            continue
        mean = sum(i.elo for i in items) / len(items)
        x = [i.elo - mean for i in items]
        centered += x
        strata.append((x, [i.correct for i in items]))
    sd = float(np.std(centered))
    strata = [([v / sd for v in x], y) for x, y in strata]

    def loglike(beta):
        total = 0.0
        for x, y in strata:
            k = sum(y)
            observed = sum(xi for xi, yi in zip(x, y) if yi)
            denom = sum(math.exp(beta * sum(x[i] for i in s))
                        for s in itertools.combinations(range(len(x)), k))
            total += beta * observed - math.log(denom)
        return total

    res = minimize_scalar(lambda b: -loglike(b), bounds=(-20, 20), method="bounded",
                          options={"xatol": 1e-9})
    lr = 2.0 * (loglike(res.x) - loglike(0.0))
    return float(res.x), float(chi2.sf(lr, df=1))


def test_conditional_logistic_matches_independent_brute_force_fit():
    # Moderate, non-separable signal with heterogeneous base rates and Elo
    # bands, so pooled and pooled-centered fits give different numbers.
    rows = (_question("a", 1150, [0, 1, 0, 1, 1])
            + _question("b", 1300, [0, 0, 1, 0, 1, 1])
            + _question("c", 1210, [1, 0, 0, 1, 1])
            + _question("d", 1100, [0, 0, 1, 1, 0, 1], step=25)
            + _question("e", 1250, [0, 1, 1, 1, 0])
            + _question("all_right", 1400, [1, 1, 1, 1], step=40))   # excluded from fit AND scale
    stats = concordance_stats(rows, bin_width=50, min_support=5)
    assert stats["logistic_n_questions"] == 5
    coef_ref, p_ref = _brute_force_conditional_fit(rows)
    assert stats["logistic_separated"] is False
    assert abs(stats["logistic_coef"] - coef_ref) < 1e-3 * max(1.0, abs(coef_ref))
    assert abs(stats["logistic_p"] - p_ref) < 1e-4


def test_conditional_logistic_is_invariant_to_a_question_elo_band_shift():
    # Only within-question position can matter: moving one whole question's
    # tournament up by 500 Elo must change neither the slope nor the p-value.
    pattern = [0, 0, 1, 0, 1, 1]
    base = [_question(f"q{q}", 1200, pattern, step=12) for q in range(5)]
    shifted = [rows if q != 2 else [ScoredHypothesis(r.elo + 500, r.correct, r.question_id)
                                    for r in rows]
               for q, rows in enumerate(base)]
    s0 = concordance_stats([r for rows in base for r in rows], 50, 5)
    s1 = concordance_stats([r for rows in shifted for r in rows], 50, 5)
    assert abs(s0["logistic_coef"] - s1["logistic_coef"]) < 1e-9
    assert abs(s0["logistic_p"] - s1["logistic_p"]) < 1e-9


# --- separation flag semantics ---------------------------------------------------

def test_separation_flag_requires_every_question_in_one_direction():
    rows = []
    for q in range(3):
        rows += _question(f"sep{q}", 1200, [0, 0, 0, 1, 1, 1])
    for q in range(3):
        rows += _question(f"mixed{q}", 1200, [0, 1, 0, 1, 0, 1])
    stats = concordance_stats(rows, 50, 5)
    assert stats["logistic_separated"] is False
    assert abs(stats["logistic_coef"]) < 10        # a finite MLE exists


def test_separation_flag_is_false_when_directions_disagree():
    # Strata separated in opposite directions: the likelihood is bounded and
    # the MLE finite, so the flag must be False.
    rows = []
    for q in range(3):
        rows += _question(f"up{q}", 1200, [0, 0, 0, 1, 1, 1])
    for q in range(3):
        rows += _question(f"down{q}", 1200, [1, 1, 1, 0, 0, 0])
    stats = concordance_stats(rows, 50, 5)
    assert stats["logistic_separated"] is False
    assert abs(stats["logistic_coef"]) < 10


def test_separation_flag_detects_the_negative_direction():
    rows = []
    for q in range(5):
        rows += _question(f"q{q}", 1200, [1, 1, 1, 0, 0, 0])
    stats = concordance_stats(rows, 50, 5)
    assert stats["logistic_separated"] is True
    assert stats["logistic_coef"] < 0


def test_separation_flag_counts_a_boundary_tie_as_separated():
    # Same Elo for one correct and one wrong row at the boundary: the
    # likelihood still rises without bound (quasi-complete separation).
    tied = [ScoredHypothesis(1190, False, "tie"), ScoredHypothesis(1200, False, "tie"),
            ScoredHypothesis(1200, True, "tie"), ScoredHypothesis(1210, True, "tie")]
    rows = tied
    for q in range(4):
        rows += _question(f"q{q}", 1200, [0, 0, 0, 1, 1, 1])
    assert concordance_stats(rows, 50, 5)["logistic_separated"] is True


# --- strata with no Elo variance carry no information --------------------------

def test_questions_with_no_elo_variance_are_not_informative():
    flat = [ScoredHypothesis(1200, c, "flat") for c in (False, True, False, True)]
    real = _question("real", 1200, [0, 0, 1, 0, 1, 1])
    stats = concordance_stats(flat + real, 50, 5)
    assert stats["logistic_n_questions"] == 1        # 'flat' excluded
    assert stats["logistic_n_rows"] == 6
    import math
    assert math.isnan(stats["logistic_coef"])        # one question is not enough


def test_all_flat_elo_reports_nothing_fitted():
    import math
    rows = []
    for q in range(4):
        rows += [ScoredHypothesis(1200, c, f"q{q}") for c in (False, False, True, True)]
    stats = concordance_stats(rows, 50, 5)
    assert stats["logistic_n_questions"] == 0
    assert stats["logistic_n_rows"] == 0
    assert math.isnan(stats["logistic_coef"])
    assert stats["logistic_separated"] is False


# --- the stratum is the tournament, which is the question only when each
# --- question was run once ------------------------------------------------------

def test_stratum_defaults_to_question_but_can_be_the_tournament():
    pattern = [0, 0, 1, 0, 1, 1]
    same_question_two_runs = (
        [ScoredHypothesis(r.elo, r.correct, "q", stratum="run-A") for r in _question("q", 1200, pattern)]
        + [ScoredHypothesis(r.elo + 300, r.correct, "q", stratum="run-B") for r in _question("q", 1200, pattern)]
    )
    assert concordance_stats(same_question_two_runs, 50, 5)["logistic_n_questions"] == 2
    pooled_by_question = [ScoredHypothesis(r.elo, r.correct, r.question_id) for r in same_question_two_runs]
    assert concordance_stats(pooled_by_question, 50, 5)["logistic_n_questions"] == 1
