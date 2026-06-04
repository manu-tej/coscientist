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
    assert stats["logistic_coef"] > 0       # positive log-odds per Elo point
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
