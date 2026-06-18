"""Tests for the live KG-on-CPU network-proximity generation method.

Offline, deterministic, no network/keys — asserts the real graph computation
discriminates the way the biology demands and composes with the voting engine.
"""
from __future__ import annotations

from company import engine, kg
from company.models import Stage
from company.science import candidate_fixtures, integrate_candidate, run_stage

PAH = "pulmonary arterial hypertension"


def test_proximity_orders_candidates_by_real_biology():
    score = lambda d: kg.proximity_score(d, PAH)
    # PDE5 inhibitors (direct strong module target) outrank the anti-remodeling
    # and preclinical candidates.
    assert score("sildenafil") > score("imatinib") > score("metformin")
    assert score("sildenafil") >= score("tadalafil") > score("imatinib")


def test_offmodule_and_unknown_abstain():
    assert kg.proximity_score("loratadine", PAH) is None   # H1 antihistamine — no PAH path
    assert kg.proximity_score("aspirin", PAH) is None       # not in the curated graph
    assert kg.proximity_score("sildenafil", "alzheimer") is None  # wrong disease


def test_proximity_is_deterministic():
    assert kg.proximity_score("sildenafil", PAH) == kg.proximity_score("sildenafil", PAH)


def test_annotate_overwrites_network_with_live_score():
    cands = candidate_fixtures(PAH)
    fixture_network = {c.drug: c.method_scores["network"] for c in cands}
    n = kg.annotate_network_scores(cands)
    assert n == 5                                           # all but loratadine score
    by_drug = {c.drug: c for c in cands}
    # sildenafil's network vote is now the computed proximity, not the fixture value
    assert by_drug["sildenafil"].method_scores["network"] == kg.proximity_score("sildenafil", PAH)
    assert by_drug["sildenafil"].method_scores["network"] != fixture_network["sildenafil"]
    # the off-module control abstains → 0.0, which integrate_candidate then ignores
    assert by_drug["loratadine"].method_scores["network"] == 0.0


def test_run_stage_hypotheses_uses_live_network_score():
    p = engine.add_program(engine.new_company("A", "PAH"), "PAH-1", PAH)
    p.stage = Stage.HYPOTHESES
    result = run_stage(p, cycle=1)
    lead = result.top_candidates[0]
    assert lead.drug == "sildenafil"
    assert lead.method_scores["network"] == kg.proximity_score("sildenafil", PAH)


def test_live_method_does_not_break_consensus_ordering():
    cands = candidate_fixtures(PAH)
    kg.annotate_network_scores(cands)
    ranked = sorted(cands, key=lambda c: integrate_candidate(c)[0], reverse=True)
    assert ranked[0].drug == "sildenafil"
    assert ranked[-1].drug == "loratadine"
