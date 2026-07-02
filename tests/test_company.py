"""Spine tests for the repurposing techbio org layer.

Everything is seeded and synchronous, so these assert exact reproducibility and the
gate state-machine mechanics without any LLM/network calls.
"""
from __future__ import annotations

import pytest

from company import engine, store
from company.cso import recommend_gate
from company.ledger import InsufficientCredits, Ledger
from company.models import GateDecision, ProgramStatus, Stage
from company.science import (
    candidate_fixtures,
    integrate_candidate,
    realized_pos,
    run_stage,
)


# --- ledger -----------------------------------------------------------------

def test_ledger_debits_company_and_program():
    pf = engine.new_company("Helix", "PAH", credit_budget=100.0)
    p = engine.add_program(pf, "PAH-1", "PAH")
    Ledger(pf.company).debit(p, credits=30.0, tokens=1000)
    assert pf.company.credit_spent == 30.0
    assert p.credits_spent == 30.0
    assert Ledger(pf.company).credits_remaining == 70.0


def test_ledger_hard_wall_on_credits():
    pf = engine.new_company("Helix", "PAH", credit_budget=10.0)
    p = engine.add_program(pf, "PAH-1", "PAH")
    with pytest.raises(InsufficientCredits):
        Ledger(pf.company).debit(p, credits=11.0)
    assert pf.company.credit_spent == 0.0  # rejected, nothing charged


# --- science: determinism + consensus ---------------------------------------

def test_run_stage_is_reproducible():
    p1 = engine.add_program(engine.new_company("A", "PAH"), "PAH-1", "pulmonary arterial hypertension", seed=42)
    p2 = engine.add_program(engine.new_company("B", "PAH"), "PAH-1", "pulmonary arterial hypertension", seed=42)
    r1, r2 = run_stage(p1, cycle=1), run_stage(p2, cycle=1)
    assert r1.confidence == r2.confidence
    assert [e.readout for e in r1.experiments] == [e.readout for e in r2.experiments]


def test_run_stage_golden_value_is_process_stable():
    # Pins cross-process reproducibility: must NOT depend on builtin hash() of strings
    # (randomized per-process by PYTHONHASHSEED). If this drifts run-to-run, seed
    # derivation regressed to an unstable hash.
    p = engine.add_program(engine.new_company("A", "PAH"), "g", "PAH", seed=7)
    assert run_stage(p, cycle=1).confidence == 0.686


def test_consensus_surfaces_sildenafil_for_pah():
    cands = candidate_fixtures("pulmonary arterial hypertension")
    ranked = sorted(cands, key=lambda c: integrate_candidate(c)[0], reverse=True)
    assert ranked[0].drug == "sildenafil"           # the known-correct repurposing
    assert ranked[-1].drug == "loratadine"          # the negative control sinks


def test_hypotheses_stage_sets_lead_candidate():
    p = engine.add_program(engine.new_company("A", "PAH"), "PAH-1", "pulmonary arterial hypertension")
    p.stage = Stage.HYPOTHESES
    run_stage(p, cycle=1)
    assert p.lead_candidate.startswith("sildenafil")


def test_hypotheses_stage_with_no_candidates_does_not_crash():
    # Any disease other than PAH yields zero fixtures in v1. The HYPOTHESES stage
    # must emit a well-formed low-confidence result rather than raising IndexError,
    # which would otherwise crash the whole quarter/portfolio run.
    p = engine.add_program(engine.new_company("A", "PAH"), "AD-1", "alzheimer disease")
    p.stage = Stage.HYPOTHESES
    result = run_stage(p, cycle=1)
    assert result.stage == Stage.HYPOTHESES.value
    assert result.confidence == 0.0
    assert result.method_agreement == 0.0
    assert result.top_candidates == []
    assert p.lead_candidate is None


def test_mechanism_routes_affinity_through_registry():
    p = engine.add_program(engine.new_company("A", "PAH"), "PAH-1", "pulmonary arterial hypertension")
    p.stage = Stage.MECHANISM
    p.confidence = 0.7
    result = run_stage(p, cycle=1)
    assert any(e.assay == "docking_affinity" for e in result.experiments)
    assert result.method_provenance.get("T0", 0) >= 1     # affinity resolved (T0 floor today)


def test_leadership_gates_expensive_cascade():
    from company.registry import FidelityTier, FnAdapter, build_default_registry
    # a registry whose affinity GPU provider actually returns a value
    reg = build_default_registry()
    reg.register(FnAdapter("gpu_live", "affinity", FidelityTier.T2, lambda i: 0.95, credits=25.0))
    # re-order so the live GPU provider is tried first
    reg._providers["affinity"].insert(0, reg._providers["affinity"].pop())
    p = engine.add_program(engine.new_company("A", "PAH"), "PAH-1", "PAH")
    p.stage = Stage.MECHANISM
    leader = run_stage(p, cycle=1, registry=reg, allow_expensive=True)
    trailer = run_stage(p, cycle=1, registry=reg, allow_expensive=False)
    assert leader.method_provenance.get("T2", 0) == 1     # leader earns the GPU run
    assert trailer.method_provenance.get("T2", 0) == 0     # trailer screened on T0 only


def test_is_leader_ranks_by_rnpv():
    pf = engine.new_company("A", "PAH")
    strong = engine.add_program(pf, "STRONG", "PAH", estimated_value=900.0)
    weak = engine.add_program(pf, "WEAK", "PAH", estimated_value=50.0)
    assert engine._is_leader(pf, strong) is True
    assert engine._is_leader(pf, weak) is False
    solo = engine.new_company("B", "PAH")
    only = engine.add_program(solo, "ONLY", "PAH")
    assert engine._is_leader(solo, only) is True           # a lone program always leads


def test_realized_pos_responds_to_science():
    hi = realized_pos(Stage.MECHANISM, confidence=0.9, red_flags=0)
    lo = realized_pos(Stage.MECHANISM, confidence=0.9, red_flags=3)
    assert hi > lo                                   # red flags drag PoS down
    assert 0.05 <= lo <= hi <= 0.95                  # clamped


# --- gate state machine -----------------------------------------------------

def test_kill_decision_terminates_program():
    pf = engine.new_company("A", "PAH")
    p = engine.add_program(pf, "PAH-1", "PAH")
    engine.run_program_stage(pf, p)
    engine.resolve_gate(pf, p, GateDecision.KILL)
    assert p.status is ProgramStatus.KILLED
    assert p.id not in pf.pending


def test_hold_keeps_stage_and_marks_held():
    pf = engine.new_company("A", "PAH")
    p = engine.add_program(pf, "PAH-1", "PAH")
    stage_before = p.stage
    engine.run_program_stage(pf, p)
    engine.resolve_gate(pf, p, GateDecision.HOLD)
    assert p.stage is stage_before
    assert p.status is ProgramStatus.HELD
    assert p.id not in pf.pending


def test_held_program_resumes_and_reruns_same_stage_next_quarter():
    pf = engine.new_company("A", "PAH")
    p = engine.add_program(pf, "PAH-1", "pulmonary arterial hypertension")
    engine.run_program_stage(pf, p)
    held_stage = p.stage
    engine.resolve_gate(pf, p, GateDecision.HOLD)
    assert p.status is ProgramStatus.HELD
    ran = engine.run_quarter(pf, auto=False)            # held → resumed, re-runs its stage
    assert [prog.id for prog, _, _ in ran] == [p.id]
    assert p.status is ProgramStatus.ACTIVE
    assert p.stage is held_stage                        # same stage, not advanced
    assert p.id in pf.pending                           # produces a fresh gate


def test_advance_survival_progresses_stage():
    pf = engine.new_company("A", "PAH")
    # high value + strong science → high PoS; find a seed whose roll survives
    p = engine.add_program(pf, "PAH-1", "pulmonary arterial hypertension", seed=42)
    p.stage = Stage.HYPOTHESES
    engine.run_program_stage(pf, p)
    rec = engine.resolve_gate(pf, p, GateDecision.ADVANCE)
    if rec.survived_roll:
        assert p.stage is Stage.MECHANISM
    else:
        assert p.status is ProgramStatus.KILLED
    assert p.cumulative_pos < 1.0                    # PoS always multiplied in


def test_full_quarter_loop_is_deterministic_and_charges_credits():
    def run():
        pf = engine.new_company("Helix", "pulmonary arterial hypertension", credit_budget=500.0, seed=7)
        engine.add_program(pf, "PAH-1", "pulmonary arterial hypertension", seed=7)
        for _ in range(6):
            engine.run_quarter(pf, auto=True)
        p = pf.programs[0]
        return p.status.value, p.stage.value, round(pf.company.credit_spent, 2)
    assert run() == run()                            # fully reproducible
    assert run()[2] > 0                              # credits were spent


# --- persistence ------------------------------------------------------------

def test_store_roundtrip(tmp_path):
    path = str(tmp_path / "state.json")
    pf = engine.new_company("Helix", "PAH", credit_budget=500.0)
    engine.add_program(pf, "PAH-1", "pulmonary arterial hypertension")
    engine.run_quarter(pf, auto=False)
    store.save(pf, path)
    pf2 = store.load(path)
    assert pf2.company.name == "Helix"
    assert pf2.programs[0].name == "PAH-1"
    assert set(pf2.pending) == set(pf.pending)
    assert len(pf2.experiments) == len(pf.experiments)
    # recommend_gate works on the reloaded pending result (enums reconstructed)
    p = pf2.programs[0]
    if p.id in pf2.pending:
        assert recommend_gate(p, pf2.pending[p.id]).decision in GateDecision
