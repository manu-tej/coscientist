from bench.concordance import (
    ScoredHypothesis, bucket_by_elo, per_bucket_accuracy, top1_accuracy,
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
