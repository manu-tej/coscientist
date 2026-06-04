import pytest
from core.models import AgentType
from bench.ablation import (
    ablation_variants, variant_config, paired_wilcoxon, cuped_adjust,
)


def test_ablation_variants_list():
    vs = ablation_variants()
    assert "full" in vs
    assert "no_evolution" in vs
    assert "no_tournament" in vs


def test_variant_config_weight_zero():
    cfg = variant_config("no_evolution")
    assert cfg["weight_overrides"][AgentType.EVOLUTION] == 0.0
    assert cfg["ranking_mode"] == "elo"


def test_variant_config_no_reflection_and_meta():
    assert variant_config("no_reflection")["weight_overrides"][AgentType.REFLECTION] == 0.0
    assert variant_config("no_meta_review")["weight_overrides"][AgentType.META_REVIEW] == 0.0


def test_variant_config_no_tournament_uses_absolute_mode():
    cfg = variant_config("no_tournament")
    assert cfg["ranking_mode"] == "absolute"
    assert cfg["weight_overrides"] == {}      # NOT a weight-zero (would freeze Elo)


def test_variant_config_full_is_baseline():
    cfg = variant_config("full")
    assert cfg["weight_overrides"] == {}
    assert cfg["ranking_mode"] == "elo"


def test_paired_wilcoxon_detects_shift():
    full = [0.8, 0.7, 0.9, 0.85, 0.75, 0.82, 0.88, 0.79]
    ablated = [0.5, 0.4, 0.6, 0.55, 0.45, 0.52, 0.58, 0.49]
    res = paired_wilcoxon(full, ablated)
    assert res["p_value"] < 0.05
    assert res["median_delta"] > 0


def test_cuped_adjust_reduces_variance():
    import statistics
    y = [0.9, 0.2, 0.8, 0.3, 0.85, 0.25]
    covariate = [0.85, 0.25, 0.78, 0.32, 0.80, 0.28]   # correlated base accuracy
    adj = cuped_adjust(y, covariate)
    assert statistics.pstdev(adj) <= statistics.pstdev(y) + 1e-9
    assert abs(statistics.mean(adj) - statistics.mean(y)) < 1e-9   # unbiased mean
