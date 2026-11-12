"""Tests for chem resolution + the Modal structure provider — all offline.

The Modal adapter is never invoked here: without `enable_structure` it isn't
registered, and when it is, an unreachable Modal abstains to the T0 floor. That
is the property worth pinning — a billed provider must never be reached by tests.
"""
from __future__ import annotations

from company import chem
from company.registry import FidelityTier, _rescale_cosine, build_default_registry

PAH = "pulmonary arterial hypertension"


def test_smiles_resolution():
    assert chem.smiles_for("Sildenafil")            # case-insensitive
    assert chem.smiles_for("sildenafil") == chem.smiles_for("  SILDENAFIL ")
    assert chem.smiles_for("not-a-drug") is None    # unresolved → method abstains


def test_leave_one_out_excludes_the_drug_itself():
    # scoring sildenafil for PAH must NOT use sildenafil as its own reference
    refs = chem.reference_actives("sildenafil", PAH)
    assert "sildenafil" not in refs
    assert "tadalafil" in refs


def test_single_active_disease_leaves_no_reference():
    # metformin is the only known active for T2D → nothing to compare against → abstain
    assert chem.reference_actives("metformin", "type 2 diabetes") == []


def test_reference_smiles_pairs_resolve():
    pairs = chem.reference_smiles("sildenafil", PAH)
    assert pairs and all(name and smi for name, smi in pairs)


def test_cosine_rescale_spreads_compressed_similarity():
    assert _rescale_cosine(0.45) == 0.0        # unrelated-molecule floor → 0
    assert _rescale_cosine(1.0) == 1.0
    assert _rescale_cosine(0.30) == 0.0        # clamped, never negative
    assert _rescale_cosine(0.83) > _rescale_cosine(0.67)   # ordering preserved


def test_structure_facet_defaults_to_t0_floor():
    reg = build_default_registry()             # structure NOT enabled
    assert [a.name for a in reg.providers("structure")] == ["t0_structure"]
    out = reg.resolve("structure", {"drug": "sildenafil", "indication": PAH,
                                    "fixture_score": 0.88})
    assert out.tier is FidelityTier.T0 and out.value == 0.88


def test_structure_abstains_to_floor_when_modal_unreachable(monkeypatch):
    reg = build_default_registry(enable_structure=True)
    # simulate Modal being unavailable/unauthenticated
    import company.registry as R
    monkeypatch.setattr(R, "_modal_structure", lambda i: None)
    reg._providers["structure"][0] = R.FnAdapter(
        "modal_chemberta", "structure", FidelityTier.T2, lambda i: None, credits=2.0)
    out = reg.resolve("structure", {"drug": "sildenafil", "indication": PAH,
                                    "fixture_score": 0.5})
    assert out.tier is FidelityTier.T0 and out.value == 0.5


def test_unresolvable_drug_abstains_without_calling_modal():
    from company.registry import _modal_structure
    # no SMILES → returns None before any network/Modal call is attempted
    assert _modal_structure({"drug": "not-a-drug", "indication": PAH}) is None
    # no leave-one-out reference → same
    assert _modal_structure({"drug": "metformin", "indication": "type 2 diabetes"}) is None
