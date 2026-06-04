from __future__ import annotations

from bench.runner import BenchHypothesis

_INITIAL_ELO = 1200.0


def elo_as_of(trajectory: list[tuple[str, float]], boundary_ts: str) -> float:
    """The hypothesis's Elo as of boundary_ts: the last trajectory point whose
    timestamp is <= boundary_ts, or the initial 1200 if none. Timestamps compare
    lexically (ISO-8601 / sortable), matching SQLite CURRENT_TIMESTAMP ordering."""
    elo = _INITIAL_ELO
    for ts, value in trajectory:
        if ts <= boundary_ts:
            elo = value
        else:
            break
    return elo


def temporal_buckets(
    hypotheses: list[BenchHypothesis], n_buckets: int = 10, mode: str = "time"
) -> list[list[BenchHypothesis]]:
    """Partition hypotheses into n_buckets by created_at.
    mode='time': equal-width time slices; mode='count': equal-count slices."""
    ordered = sorted(hypotheses, key=lambda h: h.created_at)
    if not ordered:
        return [[] for _ in range(n_buckets)]

    if mode == "count":
        buckets: list[list[BenchHypothesis]] = [[] for _ in range(n_buckets)]
        for i, h in enumerate(ordered):
            idx = min(i * n_buckets // len(ordered), n_buckets - 1)
            buckets[idx].append(h)
        return buckets

    lo, hi = ordered[0].created_at, ordered[-1].created_at
    if lo == hi:
        return [list(ordered)] + [[] for _ in range(n_buckets - 1)]
    buckets = [[] for _ in range(n_buckets)]
    n = len(ordered)
    for i, h in enumerate(ordered):
        idx = min(i * n_buckets // n, n_buckets - 1)
        buckets[idx].append(h)
    return buckets


def scaling_curve(
    hypotheses: list[BenchHypothesis], n_buckets: int = 10, mode: str = "time"
) -> list[dict]:
    """Per cumulative time bucket, best Elo and top-10-avg Elo, using each
    hypothesis's Elo as of the bucket's right boundary (no future leakage)."""
    buckets = temporal_buckets(hypotheses, n_buckets, mode)
    curve: list[dict] = []
    cumulative: list[BenchHypothesis] = []
    for b_idx, bucket in enumerate(buckets):
        cumulative = cumulative + bucket
        if not cumulative:
            curve.append({"bucket": b_idx + 1, "best_elo": _INITIAL_ELO,
                          "top10_avg_elo": _INITIAL_ELO, "n": 0})
            continue
        boundary = bucket[-1].created_at if bucket else cumulative[-1].created_at
        elos = [elo_as_of(h.elo_trajectory, boundary) for h in cumulative]
        elos.sort(reverse=True)
        top10 = elos[:10]
        curve.append({
            "bucket": b_idx + 1,
            "best_elo": elos[0],
            "top10_avg_elo": sum(top10) / len(top10),
            "n": len(cumulative),
        })
    return curve


def scaling_monotonicity(curve: list[dict], metric: str = "best_elo") -> dict:
    """Spearman ρ(bucket, metric) and OLS slope; 'no saturation' = positive slope
    over the last 3 buckets."""
    from scipy.stats import spearmanr
    xs = [pt["bucket"] for pt in curve]
    ys = [pt[metric] for pt in curve]
    if len(set(ys)) < 2:
        rho, p = float("nan"), float("nan")
    else:
        rho, p = spearmanr(xs, ys)
    tail = curve[-3:] if len(curve) >= 3 else curve
    tail_slope = (tail[-1][metric] - tail[0][metric]) if len(tail) >= 2 else 0.0
    return {"spearman_rho": float(rho), "spearman_p": float(p),
            "tail_slope": tail_slope, "no_saturation": tail_slope > 0}
