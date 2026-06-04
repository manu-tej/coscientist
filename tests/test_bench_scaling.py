from bench.runner import BenchHypothesis
from bench.scaling import elo_as_of, temporal_buckets, scaling_curve


def _bh(hid, created_at, traj):
    return BenchHypothesis(id=hid, text="", summary="", elo_rating=traj[-1][1] if traj else 1200.0,
                           created_at=created_at, elo_trajectory=traj)


def test_elo_as_of_uses_last_point_before_boundary():
    traj = [("t1", 1216.0), ("t3", 1240.0), ("t5", 1260.0)]
    assert elo_as_of(traj, "t0") == 1200.0       # before any match → initial
    assert elo_as_of(traj, "t3") == 1240.0       # inclusive of boundary
    assert elo_as_of(traj, "t4") == 1240.0       # last point <= t4
    assert elo_as_of(traj, "t9") == 1260.0


def test_temporal_buckets_equal_count_and_time():
    hyps = [_bh(f"h{i}", f"t{i:02d}", [(f"t{i:02d}", 1200.0 + i)]) for i in range(10)]
    by_time = temporal_buckets(hyps, n_buckets=5, mode="time")
    assert len(by_time) == 5
    by_count = temporal_buckets(hyps, n_buckets=5, mode="count")
    assert all(len(b) == 2 for b in by_count)


def test_scaling_curve_monotone():
    hyps = []
    for i in range(10):
        ts = f"t{i:02d}"
        hyps.append(_bh(f"h{i}", ts, [(ts, 1200.0 + i * 10)]))
    curve = scaling_curve(hyps, n_buckets=5, mode="count")
    best = [pt["best_elo"] for pt in curve]
    assert best == sorted(best)        # non-decreasing
    assert curve[0]["bucket"] == 1
    assert "top10_avg_elo" in curve[0]
