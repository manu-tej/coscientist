"""Tests for the model registry: fidelity fallback + cost-tiered cascade."""
from __future__ import annotations

import pytest

from company import kg
from company.registry import (
    FidelityTier,
    FnAdapter,
    ModelRegistry,
    RegistryError,
    build_default_registry,
)
from company.science import candidate_fixtures, integrate_candidate

PAH = "pulmonary arterial hypertension"


def test_real_provider_wins_when_it_can_speak():
    reg = build_default_registry()
    out = reg.resolve("network_proximity",
                      {"drug": "sildenafil", "indication": PAH, "fixture_score": 0.84})
    assert out.tier is FidelityTier.T1
    assert out.provider == "kg_cpu"
    assert out.value == kg.proximity_score("sildenafil", PAH)


def test_falls_through_to_t0_floor_on_abstain():
    reg = build_default_registry()
    # loratadine has no KG path → T1 abstains → T0 fixture estimate answers
    out = reg.resolve("network_proximity",
                      {"drug": "loratadine", "indication": PAH, "fixture_score": 0.15})
    assert out.tier is FidelityTier.T0
    assert out.value == 0.15


def test_unconfigured_facet_fails_loud():
    reg = ModelRegistry()
    with pytest.raises(RegistryError):
        reg.resolve("nonexistent", {})


def test_cost_tiered_cascade_gates_expensive_providers():
    reg = ModelRegistry()
    reg.register(FnAdapter("gpu", "affinity", FidelityTier.T2, lambda i: 0.99, credits=25.0))
    reg.register(FnAdapter("t0", "affinity", FidelityTier.T0, lambda i: 0.5, credits=0.0))
    # leader: expensive GPU provider allowed → it wins
    leader = reg.resolve("affinity", {}, allow_expensive=True)
    assert leader.provider == "gpu" and leader.tier is FidelityTier.T2
    # non-leader: GPU gated out → cheap T0 floor answers instead
    cheap = reg.resolve("affinity", {}, allow_expensive=False)
    assert cheap.provider == "t0" and cheap.tier is FidelityTier.T0


def test_provider_exception_does_not_sink_the_cascade():
    reg = ModelRegistry()
    def boom(_): raise RuntimeError("provider blew up")
    reg.register(FnAdapter("flaky", "x", FidelityTier.T1, boom, credits=0.0))
    reg.register(FnAdapter("floor", "x", FidelityTier.T0, lambda i: 0.3, credits=0.0))
    out = reg.resolve("x", {})
    assert out.provider == "floor" and out.value == 0.3


def test_consensus_ordering_preserved_through_registry():
    reg = build_default_registry()
    cands = candidate_fixtures(PAH)
    for c in cands:
        c.method_scores["network"] = reg.resolve(
            "network_proximity",
            {"drug": c.drug, "indication": c.indication,
             "fixture_score": c.method_scores.get("network", 0.0)}).value
    ranked = sorted(cands, key=lambda c: integrate_candidate(c)[0], reverse=True)
    assert ranked[0].drug == "sildenafil"
    assert ranked[-1].drug == "loratadine"
