"""Tests for the live KG-on-CPU network-proximity generation method.

Offline, deterministic, no network/keys — asserts the real graph computation
discriminates the way the biology demands and composes with the voting engine.
"""
from __future__ import annotations

from company import engine, kg
from company.models import Stage
from company.science import run_stage

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


def test_run_stage_hypotheses_uses_live_network_score():
    p = engine.add_program(engine.new_company("A", "PAH"), "PAH-1", PAH)
    p.stage = Stage.HYPOTHESES
    result = run_stage(p, cycle=1)
    lead = result.top_candidates[0]
    assert lead.drug == "sildenafil"
    # the lead's network vote is the real KG proximity (resolved T1 via the registry)
    assert lead.method_scores["network"] == kg.proximity_score("sildenafil", PAH)
    assert result.method_provenance.get("T1", 0) >= 1     # at least one live computation
