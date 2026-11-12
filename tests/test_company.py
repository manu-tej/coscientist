"""Spine tests for the repurposing techbio org layer.

Everything is seeded and synchronous, so these assert exact reproducibility and the
gate state-machine mechanics without any LLM/network calls.
"""
from __future__ import annotations

import json

import pytest

from company import engine, store
from company.cso import recommend_gate
from company.ledger import InsufficientCredits, InsufficientTokens, Ledger
from company.models import (
    STAGE_ORDER,
    GateDecision,
    GateStatus,
    Program,
    ProgramStatus,
    Stage,
)
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


def test_ledger_hard_wall_on_tokens_is_atomic():
    pf = engine.new_company("Helix", "PAH", token_budget=100, credit_budget=10)
    p = engine.add_program(pf, "PAH-1", "PAH")
    with pytest.raises(InsufficientTokens):
        Ledger(pf.company).debit(p, credits=2, tokens=101)
    assert pf.company.credit_spent == 0
    assert pf.company.token_spent == 0
    assert p.credits_spent == 0
    assert p.tokens_spent == 0


@pytest.mark.parametrize("credits", [-1, float("nan"), float("inf"), True])
def test_ledger_rejects_invalid_credit_debits_atomically(credits):
    pf = engine.new_company("Helix", "PAH", credit_budget=10)
    p = engine.add_program(pf, "PAH-1", "PAH")

    with pytest.raises(ValueError, match="credits"):
        Ledger(pf.company).debit(p, credits=credits)

    assert pf.company.credit_spent == 0
    assert pf.company.token_spent == 0
    assert p.credits_spent == 0
    assert p.tokens_spent == 0


@pytest.mark.parametrize("tokens", [-1, 1.5, float("nan"), float("inf"), True])
def test_ledger_rejects_invalid_token_debits_atomically(tokens):
    pf = engine.new_company("Helix", "PAH", token_budget=100)
    p = engine.add_program(pf, "PAH-1", "PAH")

    with pytest.raises(ValueError, match="tokens"):
        Ledger(pf.company).debit(p, credits=1, tokens=tokens)

    assert pf.company.credit_spent == 0
    assert pf.company.token_spent == 0
    assert p.credits_spent == 0
    assert p.tokens_spent == 0


def test_failed_stage_debit_restores_mutated_lead():
    pf = engine.new_company(
        "Helix", "PAH", token_budget=1, credit_budget=100,
    )
    p = engine.add_program(pf, "PAH-1", "pulmonary arterial hypertension")
    p.stage = Stage.HYPOTHESES
    with pytest.raises(InsufficientTokens):
        engine.run_program_stage(pf, p)
    assert p.lead_candidate is None
    assert not pf.pending
    assert not pf.stage_runs


# --- science: determinism + consensus ---------------------------------------

def test_run_stage_is_reproducible():
    p1 = engine.add_program(engine.new_company("A", "PAH"), "PAH-1", "pulmonary arterial hypertension", seed=42)
    p2 = engine.add_program(engine.new_company("B", "PAH"), "PAH-1", "pulmonary arterial hypertension", seed=42)
    r1, r2 = run_stage(p1, cycle=1), run_stage(p2, cycle=1)
    assert r1.confidence == r2.confidence
    assert [e.readout for e in r1.experiments] == [e.readout for e in r2.experiments]


def test_experiment_seed_replays_the_recorded_readout():
    import random
    from company.science import run_experiment

    confidence = 0.7
    agreement = 0.6
    experiment = run_experiment(
        random.Random(9),
        program_id="p",
        stage=Stage.MECHANISM,
        assay="assay",
        method="method",
        confidence=confidence,
        agreement=agreement,
        credits=1,
    )
    sigma = 0.28 * (1.0 - 0.6 * agreement)
    replayed = max(
        0.0,
        min(1.0, random.Random(experiment.seed).gauss(confidence, sigma)),
    )
    assert experiment.readout == round(replayed, 3)


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


def test_candidate_confidence_requires_broad_method_support():
    from company.models import Candidate

    single_vote = Candidate(
        "drug-a",
        "disease",
        {"literature": 0.9},
    )
    broad_consensus = Candidate(
        "drug-b",
        "disease",
        {method: 0.6 for method in (
            "kg", "signature", "target", "structure", "network", "literature"
        )},
    )

    single_confidence, _ = integrate_candidate(single_vote)
    broad_confidence, _ = integrate_candidate(broad_consensus)

    assert single_confidence == pytest.approx(0.15)
    assert broad_confidence == pytest.approx(0.6)
    assert broad_confidence > single_confidence


def test_hypotheses_stage_sets_lead_candidate():
    p = engine.add_program(engine.new_company("A", "PAH"), "PAH-1", "pulmonary arterial hypertension")
    p.stage = Stage.HYPOTHESES
    run_stage(p, cycle=1)
    assert p.lead_candidate.startswith("sildenafil")
    assert p.lead_drug == "sildenafil"
    assert p.lead_indication == "pulmonary arterial hypertension"


def test_structured_and_legacy_lead_fields_stay_compatible():
    p = Program(
        id="p", company_id="c", name="P", disease="D",
        lead_candidate="old drug -> old indication",
    )
    assert (p.lead_drug, p.lead_indication) == ("old drug", "old indication")

    p.lead_drug = "new drug"
    p.lead_indication = "new indication"
    assert p.lead_candidate == "new drug -> new indication"

    p.lead_candidate = None
    assert p.lead_drug is None
    assert p.lead_indication is None


def test_candidate_pool_falls_back_to_drug_library():
    from company.science import candidate_pool
    assert all(c.is_fixture for c in candidate_pool("pulmonary arterial hypertension"))
    pool = candidate_pool("multiple myeloma")
    assert pool and not any(c.is_fixture for c in pool)
    assert all(c.method_scores == {} for c in pool)     # no priors — live methods only


def test_regulatory_record_never_infers_approval_for_unknown_agent_drug():
    from company.science import regulatory_record

    known = regulatory_record("sildenafil")
    unknown = regulatory_record("agent-invented-compound")

    assert known == {
        "status": "curated_approved",
        "source": "company.science approved-drug screen v1",
        "live_verified": False,
    }
    assert unknown["status"] == "unknown"
    assert unknown["live_verified"] is False


def test_clinical_package_does_not_claim_unknown_candidate_is_approved():
    from company.models import Candidate, StageResult, StageRun

    pf = engine.new_company("Helix", "PAH")
    program = engine.add_program(pf, "PAH-1", "PAH")
    result = StageResult(
        stage=Stage.HYPOTHESES.value,
        confidence=0.5,
        method_agreement=0.2,
        red_flags=1,
        top_candidates=[
            Candidate(
                "agent-invented-compound",
                "PAH",
                {"literature": 0.9},
                "Agent-nominated candidate.",
                is_fixture=False,
            )
        ],
    )
    pf.stage_runs.append(StageRun(
        run_id="run",
        program_id=program.id,
        cycle=1,
        result=result,
    ))

    package = engine.build_clinical_hypothesis_package(pf, program)

    assert package["regulatory"]["status"] == "unknown"
    assert "unknown" in package["known_safety"]
    assert "must not be assumed" in package["known_safety"]
    assert "is an approved drug" not in package["known_safety"]


def test_uncovered_disease_abstains_instead_of_crashing():
    # No fixtures and no live method reaches myeloma offline → honest zero, no IndexError.
    p = engine.add_program(engine.new_company("A", "MM"), "MM-1", "multiple myeloma")
    p.stage = Stage.HYPOTHESES
    result = run_stage(p, cycle=1)
    assert result.confidence == 0.0
    assert result.red_flags == 1
    assert p.lead_candidate is None
    assert result.top_candidates == []


def test_live_literature_rediscovers_thalidomide_for_myeloma():
    """The disease-agnostic path: no fixtures for myeloma, so the drug-library screen
    is scored purely by the (stubbed) literature method — and the real repurposing wins."""
    from company.registry import build_default_registry
    counts = {"thalidomide": 5000, "lenalidomide": 3000, "imatinib": 200}
    def fetch(term):
        return next((n for d, n in counts.items() if f'"{d}"' in term), 3)
    reg = build_default_registry(enable_literature=True, literature_fetch=fetch)
    p = engine.add_program(engine.new_company("A", "MM"), "MM-1", "multiple myeloma")
    p.stage = Stage.HYPOTHESES
    result = run_stage(p, cycle=1, registry=reg)
    assert result.top_candidates[0].drug == "thalidomide"
    assert p.lead_candidate.startswith("thalidomide")
    assert result.method_provenance.get("T1", 0) >= 1


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


def test_provider_credits_and_fidelity_are_charged_and_provenanced():
    from company.models import STAGE_CREDIT_COST
    from company.registry import FidelityTier, FnAdapter, build_default_registry
    from company.science import estimate_stage_credits

    seen = {}

    def affinity(inputs):
        seen.update(inputs)
        return 0.91, {"raw": "computed"}

    reg = build_default_registry()
    reg.register(FnAdapter(
        "gpu_live", "affinity", FidelityTier.T2,
        affinity, credits=25.0,
    ))
    reg._providers["affinity"].insert(0, reg._providers["affinity"].pop())
    p = engine.add_program(engine.new_company("A", "PAH"), "PAH-1", "PAH")
    p.stage = Stage.MECHANISM
    p.lead_drug = "sildenafil"
    p.lead_indication = "pulmonary arterial hypertension"
    p.confidence = 0.7

    assert estimate_stage_credits(p, registry=reg) == (
        STAGE_CREDIT_COST[Stage.MECHANISM] + 25.0
    )
    result = run_stage(p, cycle=1, registry=reg, allow_expensive=True)
    docking = next(e for e in result.experiments if e.assay == "docking_affinity")
    assert result.provider_credits_spent == 25.0
    assert result.credits_spent == result.base_credits_spent + 25.0
    assert docking.fidelity_tier == "T2"
    assert docking.provider == "gpu_live"
    assert docking.evidence == {"raw": "computed"}
    assert seen["drug"] == "sildenafil"
    assert seen["indication"] == "pulmonary arterial hypertension"


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


def test_resolved_stage_results_remain_in_append_only_archive():
    pf = engine.new_company("A", "PAH")
    p = engine.add_program(pf, "PAH-1", "pulmonary arterial hypertension")

    first, _ = engine.run_program_stage(pf, p)
    first_run = pf.stage_runs[0]
    first_record = engine.resolve_gate(pf, p, GateDecision.HOLD)

    assert first_run.result is first
    assert first_run.program_id == p.id
    assert first_run.cycle == 0
    assert first_run.run_id
    assert first_run.gate_status is GateStatus.HELD
    assert first_record.run_id == first_run.run_id
    assert first_record.cycle == first_run.cycle

    engine.run_quarter(pf)
    second_run = pf.stage_runs[-1]
    engine.resolve_gate(pf, p, GateDecision.KILL)
    assert len(pf.stage_runs) == 2
    assert second_run.run_id != first_run.run_id
    assert second_run.cycle == 1
    assert second_run.gate_status is GateStatus.KILLED
    assert first_run.gate_status is GateStatus.HELD


def test_stage_cannot_rerun_while_its_gate_is_pending():
    pf = engine.new_company("A", "PAH")
    p = engine.add_program(pf, "PAH-1", "PAH")
    engine.run_program_stage(pf, p)
    with pytest.raises(ValueError, match="already has a pending gate"):
        engine.run_program_stage(pf, p)
    assert len(pf.stage_runs) == 1


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


def test_completed_package_composes_every_stage_run_and_gate_after_reload(
    tmp_path, monkeypatch,
):
    class AlwaysSurvive:
        @staticmethod
        def random():
            return 0.0

    monkeypatch.setattr(engine, "_gate_rng", lambda program, cycle: AlwaysSurvive())
    pf = engine.new_company(
        "Helix",
        "pulmonary arterial hypertension",
        credit_budget=500.0,
    )
    p = engine.add_program(
        pf,
        "PAH-ALL-STAGES",
        "pulmonary arterial hypertension",
    )

    archived_run_ids = []
    for expected_stage in STAGE_ORDER:
        assert p.stage is expected_stage
        engine.run_program_stage(pf, p)
        archived_run_ids.append(pf.stage_runs[-1].run_id)
        gate = engine.resolve_gate(pf, p, GateDecision.ADVANCE)
        assert gate.run_id == archived_run_ids[-1]

    assert p.status is ProgramStatus.COMPLETED
    state_path = tmp_path / "all-stages.json"
    store.save(pf, str(state_path))
    reloaded = store.load(str(state_path))
    p2 = reloaded.programs[0]

    package = engine.build_clinical_hypothesis_package(reloaded, p2)
    pipeline = package["pipeline"]
    stage_runs = pipeline["stage_runs"]

    assert pipeline["stage_order"] == [stage.value for stage in STAGE_ORDER]
    assert [entry["stage"] for entry in stage_runs] == pipeline["stage_order"]
    assert [entry["stage_index"] for entry in stage_runs] == list(
        range(len(STAGE_ORDER))
    )
    assert [entry["run_id"] for entry in stage_runs] == archived_run_ids
    assert all(entry["source"] == "archive" for entry in stage_runs)
    assert len(pipeline["gate_decisions"]) == len(STAGE_ORDER)
    assert all(
        decision["stage_run_linked"]
        for decision in pipeline["gate_decisions"]
    )
    assert [
        decision["run_id"] for decision in pipeline["gate_decisions"]
    ] == archived_run_ids
    assert all(
        entry["gate_decisions"][0]["run_id"] == entry["run_id"]
        for entry in stage_runs
    )

    by_stage = {entry["stage"]: entry for entry in stage_runs}
    assert {
        experiment["assay"]
        for experiment in by_stage[Stage.DISEASE_DOSSIER.value]["experiments"]
    } == {"target_enrichment", "pathway_coherence"}
    assert by_stage[Stage.HYPOTHESES.value]["candidates"][0]["drug"] == (
        package["drug"]
    )
    assert by_stage[Stage.HYPOTHESES.value]["candidates"][0][
        "evidence_graph"
    ]
    assert {
        experiment["assay"]
        for experiment in by_stage[Stage.MECHANISM.value]["experiments"]
    } == {"docking_affinity", "network_proximity"}
    assert {
        experiment["assay"]
        for experiment in by_stage[Stage.TRANSLATIONAL.value]["experiments"]
    } == {"rwe_signal", "feasibility"}
    assert all(
        "seed" in experiment
        and "fidelity_tier" in experiment
        and "evidence" in experiment
        for entry in stage_runs
        for experiment in entry["experiments"]
    )


def test_completed_program_package_uses_archived_hypotheses_after_reload(
    tmp_path, monkeypatch, capsys,
):
    class AlwaysSurvive:
        @staticmethod
        def random():
            return 0.0

    monkeypatch.setattr(engine, "_gate_rng", lambda program, cycle: AlwaysSurvive())
    pf = engine.new_company("Helix", "PAH", credit_budget=500.0)
    p = engine.add_program(pf, "PAH-1", "pulmonary arterial hypertension")

    p.stage = Stage.HYPOTHESES
    hypotheses, _ = engine.run_program_stage(pf, p)
    engine.resolve_gate(pf, p, GateDecision.ADVANCE)

    # Skip directly to the final gate; the package must still select the latest
    # hypotheses result rather than the current/translational result.
    p.stage = Stage.TRANSLATIONAL
    engine.run_program_stage(pf, p)
    engine.resolve_gate(pf, p, GateDecision.ADVANCE)
    assert p.status is ProgramStatus.COMPLETED
    assert not pf.pending

    path = str(tmp_path / "completed.json")
    store.save(pf, path)
    reloaded = store.load(path)
    p2 = reloaded.programs[0]
    package = engine.build_clinical_hypothesis_package(reloaded, p2)

    assert p2.status is ProgramStatus.COMPLETED
    assert package["drug"] == hypotheses.top_candidates[0].drug == "sildenafil"
    assert package["indication"] == p2.lead_indication
    assert package["program"]["stage"] == Stage.COMPLETED.value
    assert reloaded.latest_stage_run(p2, Stage.HYPOTHESES).gate_status is GateStatus.ADVANCED
    assert [
        entry["stage"] for entry in package["pipeline"]["stage_runs"]
    ] == [
        Stage.HYPOTHESES.value,
        Stage.TRANSLATIONAL.value,
    ]
    assert all(
        entry["gate_decisions"]
        for entry in package["pipeline"]["stage_runs"]
    )

    from company.cli import main
    main(["--state", path, "package", p2.id, "--json"])
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["drug"] == "sildenafil"
    assert emitted["program"]["stage"] == Stage.COMPLETED.value
    assert emitted["pipeline"] == package["pipeline"]


def test_package_surfaces_legacy_pending_result_without_inventing_run_or_gate():
    pf = engine.new_company("Helix", "PAH")
    p = engine.add_program(
        pf,
        "PAH-LEGACY",
        "pulmonary arterial hypertension",
    )
    p.stage = Stage.HYPOTHESES
    pending = run_stage(p, cycle=1)
    pf.pending[p.id] = pending

    package = engine.build_clinical_hypothesis_package(pf, p)
    pipeline = package["pipeline"]

    assert package["hierarchy"]["stage_run_id"] == ""
    assert len(pipeline["stage_runs"]) == 1
    assert pipeline["stage_runs"][0]["stage"] == Stage.HYPOTHESES.value
    assert pipeline["stage_runs"][0]["source"] == "legacy_pending"
    assert pipeline["stage_runs"][0]["run_id"] == ""
    assert pipeline["stage_runs"][0]["gate_status"] == GateStatus.PENDING.value
    assert pipeline["stage_runs"][0]["gate_decisions"] == []
    assert pipeline["gate_decisions"] == []


def test_package_keeps_repeated_stage_runs_and_does_not_guess_legacy_gate_link():
    pf = engine.new_company("Helix", "PAH")
    p = engine.add_program(
        pf,
        "PAH-REPEAT",
        "pulmonary arterial hypertension",
    )
    p.stage = Stage.HYPOTHESES

    engine.run_program_stage(pf, p)
    held = engine.resolve_gate(pf, p, GateDecision.HOLD)
    held.run_id = None  # shape of a legacy gate record without StageRun identity
    engine.run_program_stage(pf, p)

    package = engine.build_clinical_hypothesis_package(pf, p)
    pipeline = package["pipeline"]

    assert [entry["stage"] for entry in pipeline["stage_runs"]] == [
        Stage.HYPOTHESES.value,
        Stage.HYPOTHESES.value,
    ]
    assert [entry["gate_status"] for entry in pipeline["stage_runs"]] == [
        GateStatus.HELD.value,
        GateStatus.PENDING.value,
    ]
    assert pipeline["gate_decisions"][0]["run_id"] is None
    assert pipeline["gate_decisions"][0]["stage_run_linked"] is False
    assert all(
        entry["gate_decisions"] == []
        for entry in pipeline["stage_runs"]
    )


def test_old_snapshot_loads_and_migrates_pending_result_to_stage_run(tmp_path):
    path = tmp_path / "old.json"
    pf = engine.new_company("Helix", "PAH")
    p = engine.add_program(pf, "PAH-1", "pulmonary arterial hypertension")
    p.stage = Stage.HYPOTHESES
    engine.run_program_stage(pf, p)
    store.save(pf, str(path))

    payload = json.loads(path.read_text())
    payload.pop("stage_runs")
    payload["programs"][0].pop("lead_drug")
    payload["programs"][0].pop("lead_indication")
    path.write_text(json.dumps(payload))

    migrated = store.load(str(path))
    p2 = migrated.programs[0]
    run = migrated.latest_stage_run(p2, Stage.HYPOTHESES)
    assert p2.lead_drug == "sildenafil"
    assert p2.lead_candidate.startswith("sildenafil ->")
    assert run is not None
    assert run.run_id.startswith(f"legacy-{p2.id}-")
    assert run.gate_status is GateStatus.PENDING
