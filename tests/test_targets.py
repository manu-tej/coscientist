"""Tests for target-protein resolution and the Boltz-2 affinity provider.

Offline: the domain-selection logic is a pure function, and every abstain path is
exercised without touching UniProt or Modal. A billed provider is never reached.
"""
from __future__ import annotations

from company import targets
from company.registry import FidelityTier, _modal_boltz2, build_default_registry

PAH = "pulmonary arterial hypertension"


def _feat(kind, start, end=None):
    return {"type": kind, "location": {"start": {"value": start},
                                       "end": {"value": end if end else start}}}


def test_target_mapping():
    assert targets.target_for(PAH).gene == "PDE5A"
    assert targets.target_for("  MULTIPLE MYELOMA ").gene == "CRBN"   # normalized
    assert targets.target_for("scurvy") is None


def test_binding_domain_picks_the_pocket_not_the_biggest():
    # a big domain with no sites, and a smaller one containing every binding site
    feats = [_feat("Domain", 1, 500), _feat("Domain", 600, 700),
             _feat("Active site", 650), _feat("Binding site", 660), _feat("Binding site", 690)]
    assert targets._binding_domain(feats) == (600, 700)


def test_binding_domain_falls_back_to_largest_without_site_annotations():
    feats = [_feat("Domain", 1, 100), _feat("Domain", 200, 500)]
    assert targets._binding_domain(feats) == (200, 500)


def test_binding_domain_none_when_no_domains():
    assert targets._binding_domain([_feat("Binding site", 42)]) is None
    assert targets._binding_domain([]) is None


def test_unmapped_disease_resolves_to_none_without_network():
    assert targets.sequence_for("scurvy") is None


# --- affinity provider ------------------------------------------------------

def test_affinity_facet_defaults_to_t0_floor():
    reg = build_default_registry()                  # affinity NOT enabled
    assert [a.name for a in reg.providers("affinity")] == ["t0_affinity"]
    out = reg.resolve("affinity", {"prior": 0.42})
    assert out.tier is FidelityTier.T0 and out.value == 0.42


def test_boltz2_is_gated_behind_the_expensive_cascade():
    reg = build_default_registry(enable_affinity=True)
    boltz = next(a for a in reg.providers("affinity") if a.name == "modal_boltz2")
    assert boltz.credits > 10.0                     # above the cascade threshold
    # a trailing (non-leader) program must not reach the billed provider
    out = reg.resolve("affinity", {"prior": 0.3, "drug": "sildenafil", "indication": PAH},
                      allow_expensive=False)
    assert out.provider == "t0_affinity" and out.tier is FidelityTier.T0


def test_potency_score_is_monotonic_in_predicted_ic50():
    from company.registry import _potency_score
    # lower predicted log10(IC50) == tighter binding == higher score
    assert _potency_score(-2.0) > _potency_score(0.0) > _potency_score(2.0)
    assert 0.0 <= _potency_score(6.0) < 0.01          # millimolar → ~0
    assert 0.99 < _potency_score(-3.0) <= 1.0         # sub-nanomolar → ~1


def test_potency_ranks_real_pde5_inhibitors_above_controls():
    """Regression guard on MEASURED Boltz-2 output (MSA mode) for the PAH pool.

    Pins the finding that drove using the regression head: the two real PDE5
    inhibitors must outrank the loratadine negative control and metformin.
    """
    from company.registry import _potency_score
    measured = {"sildenafil": -2.187, "tadalafil": -1.796, "spironolactone": -0.851,
                "imatinib": -0.439, "loratadine": 0.486, "metformin": 1.863}
    scored = {d: _potency_score(v) for d, v in measured.items()}
    ranked = sorted(scored, key=scored.get, reverse=True)
    assert ranked[:2] == ["sildenafil", "tadalafil"]
    assert ranked[-1] == "metformin"
    assert scored["tadalafil"] > scored["loratadine"]


def test_potency_centre_keeps_msa_values_discriminative():
    """The centre must not saturate on accurate (MSA) predictions.

    A 10 uM centre collapsed the top four into a 0.034 band — unusable for ranking.
    The drug-potency centre must keep them spread.
    """
    from company.registry import _potency_score
    top4 = [-2.187, -1.796, -0.851, -0.439]     # sildenafil..imatinib, MSA mode
    spread = _potency_score(top4[0]) - _potency_score(top4[-1])
    assert spread > 0.5, f"potency scores saturated (spread {spread})"


def test_single_sequence_values_are_known_to_run_weak():
    """Same ligands, single-sequence mode, predicted ~2 log units weaker.

    Documents why the transform centre assumes MSA input: scoring these against the
    drug-potency centre understates them, which is correct — they ARE unreliable.
    """
    from company.registry import _potency_score
    seq_only = {"sildenafil": -0.5456, "tadalafil": 0.9747, "metformin": 2.4502}
    scored = {d: _potency_score(v) for d, v in seq_only.items()}
    assert scored["sildenafil"] < 0.5           # true ~3.5nM would score >0.9
    assert scored["sildenafil"] > scored["tadalafil"] > scored["metformin"]  # order holds


def test_boltz2_abstains_before_calling_modal_when_inputs_unresolved(monkeypatch):
    fetched = []
    monkeypatch.setattr(targets, "_fetch_json", lambda acc: fetched.append(acc) or {})
    # no SMILES for the drug → abstain without any remote call
    assert _modal_boltz2({"drug": "not-a-drug", "indication": PAH}) is None
    # no target mapped for the disease → likewise
    assert _modal_boltz2({"drug": "sildenafil", "indication": "scurvy"}) is None
    assert fetched == []  # UniProt was never asked
