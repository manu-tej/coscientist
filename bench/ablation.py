from __future__ import annotations

from core.models import AgentType

_WEIGHT_ZERO = {
    "no_evolution": AgentType.EVOLUTION,
    "no_meta_review": AgentType.META_REVIEW,
    "no_reflection": AgentType.REFLECTION,
}


def ablation_variants() -> list[str]:
    """All ablation variants, including the special no_tournament."""
    return ["full", "no_evolution", "no_meta_review", "no_reflection", "no_tournament"]


def variant_config(variant: str) -> dict:
    """Map a variant name to runner kwargs: weight_overrides + ranking_mode.

    - full: no changes.
    - no_<agent>: force that agent's sampling weight to 0.0 (never dispatched).
    - no_tournament: keep ranking dispatched but switch to absolute judge-score
      sort (zeroing RANKING would freeze all Elo at 1200 and break ranking, §10).
    """
    if variant == "full":
        return {"weight_overrides": {}, "ranking_mode": "elo"}
    if variant == "no_tournament":
        return {"weight_overrides": {}, "ranking_mode": "absolute"}
    if variant in _WEIGHT_ZERO:
        return {"weight_overrides": {_WEIGHT_ZERO[variant]: 0.0}, "ranking_mode": "elo"}
    raise ValueError(f"Unknown ablation variant: {variant!r}")


def paired_wilcoxon(full_scores: list[float], ablated_scores: list[float]) -> dict:
    """Wilcoxon signed-rank on paired per-goal scores (full vs ablated).
    Returns p-value, median paired delta, and n."""
    from scipy.stats import wilcoxon
    import statistics

    deltas = [f - a for f, a in zip(full_scores, ablated_scores)]
    nonzero = [d for d in deltas if d != 0]
    if not nonzero:
        return {"p_value": float("nan"), "median_delta": 0.0, "n": len(deltas)}
    try:
        stat, p = wilcoxon(full_scores, ablated_scores)
    except ValueError:
        p = float("nan")
    return {"p_value": float(p), "median_delta": statistics.median(deltas),
            "n": len(deltas)}


def cuped_adjust(y: list[float], covariate: list[float]) -> list[float]:
    """CUPED variance reduction: Y_adj = Y - θ(C - E[C]), θ = cov(Y,C)/var(C).
    Preserves the mean while shrinking variance when C correlates with Y (§17.1)."""
    import statistics

    n = len(y)
    if n < 2:
        return list(y)
    c_mean = statistics.mean(covariate)
    var_c = statistics.pvariance(covariate)
    if var_c == 0:
        return list(y)
    y_mean = statistics.mean(y)
    cov = sum((y[i] - y_mean) * (covariate[i] - c_mean) for i in range(n)) / n
    theta = cov / var_c
    return [y[i] - theta * (covariate[i] - c_mean) for i in range(n)]
