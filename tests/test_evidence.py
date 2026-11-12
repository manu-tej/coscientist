"""Tests for the typed evidence model / provenance graph (spec §10)."""
from __future__ import annotations

from company import engine, store
from company.evidence import CITATION, KG_PATH, EvidenceGraph
from company.models import Stage
from company.registry import build_default_registry
from company.science import run_stage

PAH = "pulmonary arterial hypertension"


def _pah_result(registry=None):
    p = engine.add_program(engine.new_company("A", "PAH"), "PAH-1", PAH)
    p.stage = Stage.HYPOTHESES
    return run_stage(p, cycle=1, registry=registry)


def test_kg_path_becomes_traceable_mechanism_evidence():
    lead = _pah_result().top_candidates[0]
    paths = lead.evidence.of_kind(KG_PATH)
    assert paths, "the KG mechanism path must be recorded as evidence"
    # the real, walkable mechanism: sildenafil → PDE5A → PAH
    assert "sildenafil" in paths[0].label and "PDE5A" in paths[0].label and PAH in paths[0].label


def test_every_vote_is_attributed_to_a_provider_and_tier():
    lead = _pah_result().top_candidates[0]
    scored = [e for e in lead.evidence.edges if e.relation == "scored_by"]
    assert len(scored) == 3                       # network + literature + structure
    assert all(e.produced_by and e.tier for e in scored)
    assert {e.tier for e in scored} <= {"T0", "T1", "T2"}


def test_literature_citation_carries_the_raw_count():
    reg = build_default_registry(enable_literature=True, literature_fetch=lambda term: 897)
    lead = _pah_result(registry=reg).top_candidates[0]
    cites = lead.evidence.of_kind(CITATION)
    assert cites and "897" in cites[0].label
    assert "[tiab]" in cites[0].detail          # the exact query is retained


def test_real_evidence_excludes_t0_estimates():
    # offline: literature falls to the T0 fixture floor, which produces no evidence node
    lead = _pah_result().top_candidates[0]
    kinds = {n.kind for n in lead.evidence.real_evidence}
    assert kinds == {KG_PATH}                    # only the computed KG path counts


def test_explain_renders_a_readable_chain():
    lead = _pah_result().top_candidates[0]
    lines = lead.evidence.explain()
    assert any("network" in l and "T1" in l for l in lines)
    assert any("PDE5A" in l for l in lines)      # the mechanism shows up in the chain


def test_graph_roundtrips_through_dict():
    g = _pah_result().top_candidates[0].evidence
    back = EvidenceGraph.from_dict(g.to_dict())
    assert [vars(n) for n in back.nodes] == [vars(n) for n in g.nodes]
    assert [vars(e) for e in back.edges] == [vars(e) for e in g.edges]


def test_evidence_survives_portfolio_persistence(tmp_path):
    path = str(tmp_path / "state.json")
    pf = engine.new_company("Helix", PAH, credit_budget=500.0)
    p = engine.add_program(pf, "PAH-1", PAH)
    p.stage = Stage.HYPOTHESES
    engine.run_program_stage(pf, p)
    store.save(pf, path)
    lead = store.load(path).pending[p.id].top_candidates[0]
    assert isinstance(lead.evidence, EvidenceGraph)
    assert lead.evidence.of_kind(KG_PATH)        # mechanism path survived the round-trip
