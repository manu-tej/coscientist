"""The company engine — program/gate state machine with stochastic attrition.

A `Portfolio` holds the company, its programs, pending gate packets, and the audit
trail (experiments + gate records). The engine advances simulated quarters: run
each active program's current stage (debiting the ledger once), produce a gate
packet, and apply the CEO's decision. On ADVANCE a seeded dice-roll against the
realized PoS can still kill the program through a seeded probability draw. The
draw is reproducible and informed by computed confidence; it is not itself an
experimental readout.
"""
from __future__ import annotations

import asyncio
import copy
import math
import random
import uuid
from dataclasses import dataclass, field
from numbers import Integral, Real

from company.cso import GateRecommendation, recommend_gate
from company.ledger import InsufficientCredits, InsufficientTokens, Ledger
from company.models import (
    STAGE_ORDER,
    Company,
    GateDecision,
    GateRecord,
    GateStatus,
    Program,
    ProgramStatus,
    Stage,
    StageResult,
    StageRun,
    next_stage,
)
from company.science import (
    estimate_stage_credits,
    regulatory_record,
    realized_pos,
    rnpv_contribution,
    run_stage,
)


def _nonnegative_real(value, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{field_name} must be a finite non-negative real number")
    return float(value)


def _nonnegative_int(value, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(value)


@dataclass
class Portfolio:
    company: Company
    programs: list[Program] = field(default_factory=list)
    pending: dict[str, StageResult] = field(default_factory=dict)   # program_id -> result awaiting a gate
    gate_records: list[GateRecord] = field(default_factory=list)
    experiments: list = field(default_factory=list)                 # full Experiment audit trail
    stage_runs: list[StageRun] = field(default_factory=list)        # append-only result archive

    def __post_init__(self) -> None:
        _nonnegative_int(
            self.company.token_budget,
            field_name="token_budget",
        )
        _nonnegative_real(
            self.company.credit_budget,
            field_name="credit_budget",
        )
        _nonnegative_int(
            self.company.token_spent,
            field_name="token_spent",
        )
        _nonnegative_real(
            self.company.credit_spent,
            field_name="credit_spent",
        )
        _nonnegative_int(
            self.company.failed_token_spent,
            field_name="failed_token_spent",
        )
        _nonnegative_real(
            self.company.failed_credit_spent,
            field_name="failed_credit_spent",
        )
        seen_names: set[str] = set()
        for program in self.programs:
            _nonnegative_int(
                program.tokens_spent,
                field_name="tokens_spent",
            )
            _nonnegative_real(
                program.credits_spent,
                field_name="credits_spent",
            )
            _nonnegative_real(
                program.estimated_value,
                field_name="estimated_value",
            )
            if program.name in seen_names:
                raise ValueError(
                    f"duplicate program name {program.name!r}; "
                    "program names must be unique within a portfolio"
                )
            seen_names.add(program.name)

    def program(self, pid: str) -> Program:
        for p in self.programs:
            if p.id == pid or p.name == pid:
                return p
        raise KeyError(f"no program {pid!r}")

    def runnable_programs(self) -> list[Program]:
        """Programs that should run a stage this quarter: ACTIVE, plus HELD ones the
        CEO paused for a de-risking re-run (resumed in run_quarter)."""
        return [p for p in self.programs
                if p.status in (ProgramStatus.ACTIVE, ProgramStatus.HELD)]

    def latest_stage_run(self, program: Program | str,
                         stage: Stage | str | None = None) -> StageRun | None:
        """Return the newest archived run for a program, optionally at one stage."""
        pid = program.id if isinstance(program, Program) else self.program(program).id
        stage_value = stage.value if isinstance(stage, Stage) else stage
        for run in reversed(self.stage_runs):
            if run.program_id == pid and (stage_value is None or run.result.stage == stage_value):
                return run
        return None


def new_company(name: str, disease: str, *, token_budget: int = 5_000_000,
                credit_budget: float = 500.0, seed: int = 0) -> Portfolio:
    token_budget = _nonnegative_int(token_budget, field_name="token_budget")
    credit_budget = _nonnegative_real(credit_budget, field_name="credit_budget")
    company = Company(id=str(uuid.uuid4())[:8], name=name, disease_focus=disease,
                      token_budget=token_budget, credit_budget=credit_budget, seed=seed)
    return Portfolio(company=company)


def add_program(pf: Portfolio, name: str, disease: str, *, estimated_value: float = 400.0,
                seed: int | None = None) -> Program:
    estimated_value = _nonnegative_real(
        estimated_value,
        field_name="estimated_value",
    )
    if any(program.name == name for program in pf.programs):
        raise ValueError(
            f"duplicate program name {name!r}; "
            "program names must be unique within a portfolio"
        )
    if seed is None:
        seed = pf.company.seed + len(pf.programs) + 1
    p = Program(id=str(uuid.uuid4())[:8], company_id=pf.company.id, name=name,
                disease=disease, estimated_value=estimated_value, seed=seed)
    p.history.append(f"founded @ cycle {pf.company.cycle}; est. value {estimated_value:.0f}")
    pf.programs.append(p)
    return p


def can_afford_stage(pf: Portfolio, program: Program, *, registry=None) -> bool:
    """Can the company cover the conservative stage/provider reservation?"""
    return Ledger(pf.company).can_afford(estimate_stage_credits(
        program,
        registry=registry,
        allow_expensive=_is_leader(pf, program),
    ))


def _is_leader(pf: Portfolio, program: Program) -> bool:
    """Is this program a portfolio leader (top-half by rNPV)? Leaders earn the
    expensive GPU cascade; trailing programs are screened with cheap methods only."""
    runnable = [
        candidate for candidate in pf.runnable_programs()
        if candidate.id not in pf.pending
    ]
    if len(runnable) <= 1:
        return True
    ranked = sorted(runnable, key=rnpv_contribution, reverse=True)
    cutoff = (len(ranked) + 1) // 2          # top half, rounded up
    return program in ranked[:cutoff]


def run_program_stage(pf: Portfolio, program: Program,
                      *, registry=None) -> tuple[StageResult, GateRecommendation]:
    """Run the program's current stage once, debit the ledger, stash a pending gate."""
    if program.id in pf.pending:
        raise ValueError(
            f"program {program.name!r} already has a pending gate; resolve it before re-running")
    reservation = estimate_stage_credits(
        program, registry=registry, allow_expensive=_is_leader(pf, program),
    )
    if not Ledger(pf.company).can_afford(reservation):
        raise InsufficientCredits(
            f"program {program.id!r} needs up to {reservation:.1f} credits; "
            f"only {Ledger(pf.company).credits_remaining:.1f} remain"
        )
    lead_snapshot = (
        program.lead_candidate, program.lead_drug, program.lead_indication,
    )
    try:
        result = run_stage(program, pf.company.cycle, registry=registry,
                           allow_expensive=_is_leader(pf, program))
        return _record_stage_result(pf, program, result)
    except BaseException:
        object.__setattr__(program, "lead_candidate", lead_snapshot[0])
        object.__setattr__(program, "lead_drug", lead_snapshot[1])
        object.__setattr__(program, "lead_indication", lead_snapshot[2])
        raise


def _record_stage_result(
    pf: Portfolio, program: Program, result: StageResult,
) -> tuple[StageResult, GateRecommendation]:
    """Commit one completed executor result to budgets, gates, and the archive."""
    result = _validate_stage_result(program, result)
    ledger = Ledger(pf.company)
    ledger.debit(program, credits=result.credits_spent, tokens=result.tokens_spent)
    program.confidence = result.confidence
    program.red_flags = result.red_flags
    pf.experiments.extend(result.experiments)
    pf.pending[program.id] = result
    pf.stage_runs.append(StageRun(
        run_id=str(uuid.uuid4()),
        program_id=program.id,
        cycle=pf.company.cycle,
        result=result,
    ))
    rec = recommend_gate(program, result)
    program.history.append(
        f"cycle {pf.company.cycle}: ran {program.stage.value} → conf {result.confidence:.2f}, "
        f"{result.red_flags} flags, spent {result.credits_spent:.1f}cr; CSO: {rec.decision.value}"
    )
    return result, rec


def _create_executor_token_budget(executor, company: Company):
    """Create one pre-dispatch budget when an executor supports call admission."""
    factory = getattr(executor, "create_token_budget", None)
    if not callable(factory):
        return None
    return factory(max(0, Ledger(company).tokens_remaining))


def _attempt_credit_value(value) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value < 0
    ):
        return None
    return float(value)


def _attempt_token_value(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        return None
    return int(value)


def _attempt_usage(
    settled: list[object],
    *,
    token_budget=None,
) -> tuple[float, int]:
    """Measure usage from completed results, failures, and a shared call meter."""
    credits = 0.0
    tokens = 0
    for item in settled:
        source = item
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[1], StageResult)
        ):
            source = item[1]
        item_credits = _attempt_credit_value(
            getattr(source, "attempted_credits", None)
        )
        result_credits = _attempt_credit_value(
            getattr(source, "credits_spent", None)
        )
        credits += max(item_credits or 0.0, result_credits or 0.0)

        item_tokens = _attempt_token_value(
            getattr(source, "attempted_tokens", None)
        )
        result_tokens = _attempt_token_value(
            getattr(source, "tokens_spent", None)
        )
        tokens += max(item_tokens or 0, result_tokens or 0)

    shared_spent = _attempt_token_value(getattr(token_budget, "spent", None))
    if shared_spent is not None:
        # The shared admission meter sees failed and cancelled calls that may
        # never have produced a StageResult.
        tokens = max(tokens, shared_spent)
    return credits, tokens


def _charge_failed_attempt_usage(
    company: Company,
    settled: list[object],
    *,
    token_budget=None,
) -> None:
    credits, tokens = _attempt_usage(settled, token_budget=token_budget)
    company.failed_credit_spent += credits
    company.failed_token_spent += tokens


def _validate_stage_result(program: Program, result: object) -> StageResult:
    """Validate the gate-facing result contract before any portfolio mutation."""
    if not isinstance(result, StageResult):
        raise TypeError(
            f"stage executor for {program.name!r} returned "
            f"{type(result).__name__}, expected StageResult"
        )
    if result.stage != program.stage.value:
        raise ValueError(
            f"stage executor for {program.name!r} returned stage "
            f"{result.stage!r}, expected {program.stage.value!r}"
        )
    confidence = _nonnegative_real(
        result.confidence,
        field_name="confidence",
    )
    if confidence > 1:
        raise ValueError("confidence must be in [0, 1]")
    agreement = _nonnegative_real(
        result.method_agreement,
        field_name="method_agreement",
    )
    if agreement > 1:
        raise ValueError("method_agreement must be in [0, 1]")
    _nonnegative_int(result.red_flags, field_name="red_flags")
    _nonnegative_real(result.credits_spent, field_name="credits_spent")
    _nonnegative_int(result.tokens_spent, field_name="tokens_spent")
    if not isinstance(result.top_candidates, list):
        raise ValueError("top_candidates must be a list")
    if not isinstance(result.experiments, list):
        raise ValueError("experiments must be a list")
    return result


async def _execute_stage(
    executor,
    program: Program,
    cycle: int,
    *,
    registry,
    allow_expensive: bool,
    token_budget=None,
) -> StageResult:
    kwargs = {
        "registry": registry,
        "allow_expensive": allow_expensive,
    }
    # Preserve compatibility with injected seeded/custom executors that have no
    # LLM calls and therefore do not implement the optional budget protocol.
    if token_budget is not None:
        kwargs["token_budget"] = token_budget
    return await executor.execute(program, cycle, **kwargs)


async def run_program_stage_async(
    pf: Portfolio,
    program: Program,
    *,
    executor,
    registry=None,
) -> tuple[StageResult, GateRecommendation]:
    """Run one async stage as a scientific-state transaction."""
    if program.id in pf.pending:
        raise ValueError(
            f"program {program.name!r} already has a pending gate; resolve it before re-running")
    try:
        program_index = next(
            index
            for index, candidate in enumerate(pf.programs)
            if candidate.id == program.id
        )
    except StopIteration as exc:
        raise ValueError(
            f"program {program.id!r} is not part of this portfolio"
        ) from exc
    reservation = estimate_stage_credits(
        program, registry=registry, allow_expensive=_is_leader(pf, program),
    )
    if not Ledger(pf.company).can_afford(reservation):
        raise InsufficientCredits(
            f"program {program.id!r} needs up to {reservation:.1f} credits; "
            f"only {Ledger(pf.company).credits_remaining:.1f} remain"
        )
    working = Portfolio(
        company=copy.deepcopy(pf.company),
        programs=copy.deepcopy(pf.programs),
        pending=dict(pf.pending),
        gate_records=list(pf.gate_records),
        experiments=list(pf.experiments),
        stage_runs=list(pf.stage_runs),
    )
    working_program = working.programs[program_index]
    token_budget = _create_executor_token_budget(executor, working.company)
    try:
        result = await _execute_stage(
            executor,
            working_program,
            working.company.cycle,
            registry=registry,
            allow_expensive=_is_leader(working, working_program),
            token_budget=token_budget,
        )
        result, recommendation = _record_stage_result(
            working,
            working_program,
            result,
        )
    except BaseException as exc:
        attempted: list[object] = [exc]
        if "result" in locals():
            attempted.append(result)
        _charge_failed_attempt_usage(
            pf.company,
            attempted,
            token_budget=token_budget,
        )
        raise
    _install_portfolio_state(pf, working)
    return result, recommendation


def _gate_rng(program: Program, cycle: int) -> random.Random:
    import zlib
    stable = zlib.crc32(program.stage.value.encode())
    return random.Random((program.seed * 7919 + cycle * 104_729 + stable) & 0x7FFFFFFF)


def resolve_gate(pf: Portfolio, program: Program, decision: GateDecision) -> GateRecord:
    """Apply a CEO gate decision to a program with a pending stage result."""
    result = pf.pending.pop(program.id, None)
    if result is None:
        raise ValueError(f"program {program.name!r} has no pending gate; run a quarter first")

    stage_run = pf.latest_stage_run(program, result.stage)
    if stage_run is None or stage_run.gate_status is not GateStatus.PENDING:
        # Compatibility for portfolios assembled in memory using the old ``pending``
        # shape rather than by ``run_program_stage``.
        stage_run = StageRun(
            run_id=str(uuid.uuid4()),
            program_id=program.id,
            cycle=pf.company.cycle,
            result=result,
        )
        pf.stage_runs.append(stage_run)

    pos = realized_pos(program.stage, result.confidence, result.red_flags)
    transition = f"{program.stage.value}->{next_stage(program.stage).value}"
    rec = recommend_gate(program, result)
    record = GateRecord(
        program_id=program.id, transition=transition, confidence=result.confidence,
        realized_pos=round(pos, 3), red_flags=result.red_flags,
        rnpv_contribution=round(rnpv_contribution(program), 1),
        cso_recommendation=rec.decision.value, ceo_decision=decision.value,
        run_id=stage_run.run_id, cycle=stage_run.cycle,
    )

    if decision is GateDecision.KILL:
        program.status = ProgramStatus.KILLED
        stage_run.gate_status = GateStatus.KILLED
        record.note = "CEO killed the program"
        program.history.append(f"cycle {pf.company.cycle}: GATE killed by CEO")

    elif decision is GateDecision.HOLD:
        program.status = ProgramStatus.HELD
        stage_run.gate_status = GateStatus.HELD
        record.note = "held — will re-run the stage next quarter to de-risk"
        program.history.append(f"cycle {pf.company.cycle}: GATE held by CEO")

    else:  # ADVANCE — stochastic attrition can still kill it
        survived = _gate_rng(program, pf.company.cycle).random() < pos
        record.survived_roll = survived
        program.cumulative_pos = round(program.cumulative_pos * pos, 4)
        if not survived:
            program.status = ProgramStatus.KILLED
            stage_run.gate_status = GateStatus.ATTRITED
            record.note = f"advanced but FAILED stochastically in {program.stage.value} (PoS {pos:.2f})"
            program.history.append(
                f"cycle {pf.company.cycle}: GATE advanced → FAILED roll (PoS {pos:.2f}) — attrition")
        else:
            nxt = next_stage(program.stage)
            program.stage = nxt
            if nxt is Stage.COMPLETED:
                program.status = ProgramStatus.COMPLETED
                stage_run.gate_status = GateStatus.COMPLETED
                record.note = "cleared final gate → clinical-hypothesis package"
                program.history.append(f"cycle {pf.company.cycle}: COMPLETED — {program.lead_candidate}")
            else:
                stage_run.gate_status = GateStatus.ADVANCED
                record.note = f"advanced to {nxt.value} (survived PoS {pos:.2f})"
                program.history.append(f"cycle {pf.company.cycle}: GATE advanced → {nxt.value}")

    pf.gate_records.append(record)
    return record


def _gate_record_payload(
    record: GateRecord,
    *,
    archived_run_ids: set[str],
) -> dict:
    """Return the complete stored gate decision without inferring missing links."""
    return {
        "program_id": record.program_id,
        "transition": record.transition,
        "confidence": record.confidence,
        "realized_pos": record.realized_pos,
        "red_flags": record.red_flags,
        "rnpv_contribution": record.rnpv_contribution,
        "cso_recommendation": record.cso_recommendation,
        "ceo_decision": record.ceo_decision,
        "survived_roll": record.survived_roll,
        "note": record.note,
        "run_id": record.run_id,
        "cycle": record.cycle,
        # Older snapshots can contain gate records without a StageRun identity.
        # Surface that honestly instead of guessing which repeated stage run they
        # belonged to.
        "stage_run_linked": bool(
            record.run_id and record.run_id in archived_run_ids
        ),
    }


def _candidate_payload(candidate) -> dict:
    evidence = candidate.evidence
    return {
        "drug": candidate.drug,
        "indication": candidate.indication,
        "method_scores": dict(candidate.method_scores),
        "rationale": candidate.rationale,
        "elo_rating": candidate.elo_rating,
        "is_fixture": candidate.is_fixture,
        "evidence_graph": evidence.to_dict() if evidence is not None else {},
        "evidence_chain": evidence.explain() if evidence is not None else [],
        "real_evidence_count": (
            len(evidence.real_evidence) if evidence is not None else 0
        ),
    }


def _experiment_payload(experiment) -> dict:
    """Expose every provenance field the stored Experiment actually contains."""
    return {
        "program_id": experiment.program_id,
        "stage": experiment.stage,
        "assay": experiment.assay,
        "method": experiment.method,
        "fidelity_tier": experiment.fidelity_tier,
        "readout": experiment.readout,
        "interpretation": experiment.interpretation,
        "seed": experiment.seed,
        "credits": experiment.credits,
        "provider": experiment.provider,
        "detail": experiment.detail,
        "evidence": copy.deepcopy(experiment.evidence),
    }


def _stage_run_payload(
    run: StageRun,
    *,
    source: str,
    stage_index: int | None,
    gate_decisions: list[dict],
) -> dict:
    """Serialize one archived stage result as decision-grade package evidence."""
    result = run.result
    return {
        "stage": result.stage,
        "stage_index": stage_index,
        "run_id": run.run_id,
        "cycle": run.cycle,
        "source": source,
        "gate_status": (
            run.gate_status.value
            if isinstance(run.gate_status, GateStatus)
            else str(run.gate_status)
        ),
        "confidence": result.confidence,
        "method_agreement": result.method_agreement,
        "red_flags": result.red_flags,
        "red_flag_details": list(result.red_flag_details),
        "method_provenance": dict(result.method_provenance),
        "candidates": [
            _candidate_payload(candidate)
            for candidate in result.top_candidates
        ],
        "experiments": [
            _experiment_payload(experiment)
            for experiment in result.experiments
        ],
        "quantitative_baseline": copy.deepcopy(result.quantitative_baseline),
        "team": {
            "name": result.team_name,
            "roster": list(result.team_roster),
            "summary": result.team_summary,
            "confidence": result.team_confidence,
            "coscientist_run_id": result.coscientist_run_id,
            "coscientist_db_path": result.coscientist_db_path,
            "agent_trace": copy.deepcopy(result.agent_trace),
            "candidate_ranking": copy.deepcopy(result.team_candidate_ranking),
            "provider": result.provider,
            "models": dict(result.models),
            "workers": result.workers,
            "max_tasks": result.max_tasks,
            "seed": result.seed,
            "max_backend_concurrency": result.max_backend_concurrency,
            "llm_calls": result.llm_calls,
            "token_accounting": result.token_accounting,
        },
        "spend": {
            "credits": result.credits_spent,
            "tokens": result.tokens_spent,
            "base_credits": result.base_credits_spent,
            "provider_credits": result.provider_credits_spent,
        },
        "gate_decisions": copy.deepcopy(gate_decisions),
    }


def _program_pipeline(pf: Portfolio, program: Program) -> dict:
    """Compose every archived stage run and gate in canonical pipeline order."""
    stage_rank = {
        stage.value: index
        for index, stage in enumerate(STAGE_ORDER)
    }
    runs: list[tuple[StageRun, str, int]] = [
        (run, "archive", archive_index)
        for archive_index, run in enumerate(pf.stage_runs)
        if run.program_id == program.id
    ]

    # Very old in-memory portfolios can have a pending packet but no StageRun.
    # Include it without mutating the portfolio. Current saved portfolios already
    # migrate such packets into the archive in company.store.load().
    pending = pf.pending.get(program.id)
    if pending is not None and not any(
        run.gate_status is GateStatus.PENDING
        and run.result.stage == pending.stage
        for run, _, _ in runs
    ):
        runs.append((
            StageRun(
                run_id="",
                program_id=program.id,
                cycle=pf.company.cycle,
                result=pending,
                gate_status=GateStatus.PENDING,
            ),
            "legacy_pending",
            len(pf.stage_runs),
        ))

    runs.sort(key=lambda item: (
        stage_rank.get(item[0].result.stage, len(STAGE_ORDER)),
        item[0].cycle,
        item[2],
    ))
    archived_run_ids = {
        run.run_id for run, source, _ in runs
        if source == "archive" and run.run_id
    }

    indexed_gates = [
        (record, record_index)
        for record_index, record in enumerate(pf.gate_records)
        if record.program_id == program.id
    ]

    def gate_key(item):
        record, record_index = item
        from_stage = record.transition.partition("->")[0]
        cycle = record.cycle if record.cycle is not None else math.inf
        return (
            stage_rank.get(from_stage, len(STAGE_ORDER)),
            cycle,
            record_index,
        )

    indexed_gates.sort(key=gate_key)
    gate_payloads = [
        _gate_record_payload(
            record,
            archived_run_ids=archived_run_ids,
        )
        for record, _ in indexed_gates
    ]
    gates_by_run: dict[str, list[dict]] = {}
    for gate in gate_payloads:
        run_id = gate["run_id"]
        if run_id:
            gates_by_run.setdefault(run_id, []).append(gate)

    return {
        "stage_order": [stage.value for stage in STAGE_ORDER],
        "stage_runs": [
            _stage_run_payload(
                run,
                source=source,
                stage_index=stage_rank.get(run.result.stage),
                gate_decisions=gates_by_run.get(run.run_id, []),
            )
            for run, source, _ in runs
        ],
        # Preserve legacy/unlinked decisions too. Their false
        # ``stage_run_linked`` value is an explicit provenance warning.
        "gate_decisions": gate_payloads,
    }


def build_clinical_hypothesis_package(
    pf: Portfolio, program: Program | str,
) -> dict:
    """Build the product deliverable from the latest hypotheses-stage run.

    Looking in the durable stage-run archive (instead of only ``pending``) keeps
    the package available after later gates consume the hypotheses packet and
    after a save/load round trip.
    """
    p = program if isinstance(program, Program) else pf.program(program)
    run = pf.latest_stage_run(p, Stage.HYPOTHESES)
    result = run.result if run is not None else None

    # Accept an old in-memory Portfolio whose pending result predates StageRun.
    pending = pf.pending.get(p.id)
    archived_pending_hypotheses = any(
        archived.program_id == p.id
        and archived.result.stage == Stage.HYPOTHESES.value
        and archived.gate_status is GateStatus.PENDING
        for archived in pf.stage_runs
    )
    if (
        pending is not None
        and pending.stage == Stage.HYPOTHESES.value
        and (result is None or not archived_pending_hypotheses)
    ):
        result = pending
        run = None

    lead = result.top_candidates[0] if result and result.top_candidates else None
    if lead is None:
        raise ValueError(
            f"«{p.name}» has no scored candidate yet — run a hypotheses quarter first")

    ev = lead.evidence
    regulatory = regulatory_record(lead.drug)
    if regulatory["status"] == "curated_approved":
        known_safety = (
            f"{lead.drug} is listed in the local curated approved-drug screen. "
            "This run did not re-verify its current label, safety, or PK; review "
            "current regulator-authored labeling before any decision."
        )
        dose_validation = "verify the current approved label and dose/exposure limits"
    else:
        known_safety = (
            f"Regulatory approval and safety/PK status for {lead.drug} are unknown "
            "in this run and must not be assumed from agent-generated evidence."
        )
        dose_validation = "establish regulatory status, safety, PK, and dose feasibility"
    return {
        "drug": lead.drug,
        "indication": lead.indication,
        "mechanism": (ev.of_kind("kg_path")[0].label if ev and ev.of_kind("kg_path")
                      else lead.rationale or "not established by a computed method"),
        "confidence": result.confidence,
        "method_scores": lead.method_scores,
        "method_agreement": result.method_agreement,
        "evidence_graph": ev.to_dict() if ev else {},
        "evidence_chain": ev.explain() if ev else [],
        "real_evidence_count": len(ev.real_evidence) if ev else 0,
        "fidelity": result.method_provenance,
        "red_flag_count": result.red_flags,
        "red_flags": list(result.red_flag_details),
        "regulatory": regulatory,
        "known_safety": known_safety,
        "suggested_validation": [
            f"confirm target engagement of {lead.drug} in {lead.indication} models",
            "retrospective real-world-evidence cohort on existing prescriptions",
            dose_validation,
        ],
        "program": {
            "name": p.name,
            "stage": p.stage.value,
            "cumulative_pos": p.cumulative_pos,
        },
        # Additive, canonical all-stage composition. Existing top-level candidate
        # and hierarchy fields retain their hypotheses-stage meaning for backward
        # compatibility; the pipeline makes later dossier/mechanism/translational
        # evidence and every linked CEO gate explicit.
        "pipeline": _program_pipeline(pf, p),
        "hierarchy": {
            "stage_run_id": run.run_id if run is not None else "",
            "team_name": result.team_name,
            "team_roster": list(result.team_roster),
            "team_summary": result.team_summary,
            "team_confidence": result.team_confidence,
            "coscientist_run_id": result.coscientist_run_id,
            "coscientist_db_path": result.coscientist_db_path,
            "agent_trace": list(result.agent_trace),
            "candidate_ranking": list(result.team_candidate_ranking),
            "quantitative_baseline": dict(result.quantitative_baseline),
            "llm_calls": result.llm_calls,
            "tokens_spent": result.tokens_spent,
            "token_accounting": result.token_accounting,
            "provider": result.provider,
            "models": dict(result.models),
            "workers": result.workers,
            "max_tasks": result.max_tasks,
            "seed": result.seed,
            "max_backend_concurrency": result.max_backend_concurrency,
        },
    }


def run_quarter(pf: Portfolio, *, auto: bool = False,
                registry=None) -> list[tuple[Program, StageResult, GateRecommendation]]:
    """Advance one simulated quarter: run each active program's current stage.

    With auto=True, immediately resolve each gate by the CSO's recommendation (used
    for the reproducible demo + tests). Otherwise leaves pending gates for the CEO.
    `registry` is the model registry the stage science resolves methods through.
    """
    pf.company.cycle += 1
    out: list[tuple[Program, StageResult, GateRecommendation]] = []
    for program in pf.runnable_programs():
        if program.id in pf.pending:
            continue  # already awaiting a CEO decision
        # Credits are a hard wall: a program that can't fund its next stage is held
        # rather than crashing the quarter (checked BEFORE resuming, so a
        # funds-held program doesn't churn ACTIVE→HELD every cycle).
        if not can_afford_stage(pf, program, registry=registry):
            if program.status is not ProgramStatus.HELD:
                program.status = ProgramStatus.HELD
                program.history.append(
                    f"cycle {pf.company.cycle}: insufficient credits for "
                    f"{program.stage.value} — held awaiting funding")
            continue
        if program.status is ProgramStatus.HELD:
            program.status = ProgramStatus.ACTIVE  # resume the de-risking re-run
            program.history.append(f"cycle {pf.company.cycle}: resumed from hold")
        result, rec = run_program_stage(pf, program, registry=registry)
        out.append((program, result, rec))
        if auto:
            resolve_gate(pf, program, rec.decision)
    return out


async def run_quarter_async(
    pf: Portfolio,
    *,
    executor,
    auto: bool = False,
    registry=None,
    n_program_workers: int = 4,
    approved_allocation: dict[str, float] | None = None,
) -> list[tuple[Program, StageResult, GateRecommendation]]:
    """Run program stage teams concurrently behind a bounded company semaphore.

    ``approved_allocation`` is the CEO-approved CSO envelope. When supplied, a
    program may launch only if its envelope covers the known stage reservation.
    Reservations are selected serially before any team/model call, preventing a
    concurrent quarter from oversubscribing the global credit wall.

    The whole quarter is an in-memory transaction. Executors work on an isolated
    copy of the portfolio, all selected workers are allowed to settle, and their
    exact combined debit is validated before the new state is installed. Thus a
    failure in one team cannot leave another team's result, a cycle increment, or
    a partial ledger debit behind. Executors that implement ``create_token_budget``
    also receive one shared pre-dispatch envelope capped at the company's remaining
    portable estimated/requested token units.
    """
    if n_program_workers < 1:
        raise ValueError("n_program_workers must be >= 1")

    # Only Company and Program are mutable inputs to stage execution. Historic
    # records can be shared behind copied containers: this keeps the transaction
    # setup proportional to portfolio size instead of repeatedly cloning every
    # evidence graph accumulated over the company's lifetime.
    working = Portfolio(
        company=copy.deepcopy(pf.company),
        programs=copy.deepcopy(pf.programs),
        pending=dict(pf.pending),
        gate_records=list(pf.gate_records),
        experiments=list(pf.experiments),
        stage_runs=list(pf.stage_runs),
    )
    working.company.cycle += 1
    runnable = [
        p for p in working.runnable_programs()
        if p.id not in working.pending
    ]
    # Prioritize the science/economic signal, not insertion order.
    runnable.sort(key=rnpv_contribution, reverse=True)

    remaining = Ledger(working.company).credits_remaining
    selected: list[Program] = []
    approved_envelopes: dict[str, float] = {}
    for program in runnable:
        cost = estimate_stage_credits(
            program,
            registry=registry,
            allow_expensive=_is_leader(working, program),
        )
        envelope = (
            approved_allocation.get(program.id, 0.0)
            if approved_allocation is not None else cost
        )
        reason = ""
        if envelope + 1e-9 < cost:
            reason = (
                f"approved CSO allocation {envelope:.1f}cr does not cover "
                f"{program.stage.value} reservation {cost:.1f}cr"
            )
        elif remaining + 1e-9 < cost:
            reason = (
                f"insufficient credits for {program.stage.value} "
                f"({cost:.1f}cr required, {remaining:.1f}cr available)"
            )
        if reason:
            program.status = ProgramStatus.HELD
            program.history.append(f"cycle {working.company.cycle}: {reason} — held")
            continue
        remaining -= cost
        approved_envelopes[program.id] = envelope
        if program.status is ProgramStatus.HELD:
            program.status = ProgramStatus.ACTIVE
            program.history.append(f"cycle {working.company.cycle}: resumed from hold")
        selected.append(program)

    semaphore = asyncio.Semaphore(n_program_workers)
    # Hierarchical executors opt into one shared envelope created before any
    # program worker starts. Every nested agent/model call then competes against
    # the same remaining company-token allowance.
    shared_token_budget = _create_executor_token_budget(executor, working.company)

    outcome_missing = object()
    worker_outcomes: list[object] = [
        outcome_missing for _ in selected
    ]

    async def one(index: int, program: Program):
        try:
            async with semaphore:
                result = await _execute_stage(
                    executor,
                    program,
                    working.company.cycle,
                    registry=registry,
                    allow_expensive=_is_leader(working, program),
                    token_budget=shared_token_budget,
                )
                outcome = (program, result)
                worker_outcomes[index] = outcome
                return outcome
        except BaseException as exc:
            # Task.result()/gather synthesize a new CancelledError and discard
            # attributes attached by the executor. Preserve the original object
            # so attempted credit/token metadata survives caller cancellation.
            worker_outcomes[index] = exc
            raise

    # Keep explicit task handles so caller cancellation can cancel and drain every
    # worker before the transaction returns. This also gives failed-attempt
    # accounting a settled view of calls that completed before cancellation.
    worker_tasks = [
        asyncio.create_task(one(index, program))
        for index, program in enumerate(selected)
    ]
    aggregate = asyncio.gather(
        *worker_tasks,
        return_exceptions=True,
    )
    try:
        # return_exceptions=True waits for every selected worker. Without that, a
        # slower executor could keep mutating its program after a sibling failure
        # had already escaped to the caller.
        # Shield prevents repeated caller cancellation from being forwarded into
        # a worker while it is attaching usage metadata to its first cancellation.
        gathered = await asyncio.shield(aggregate)
        settled = [
            gathered_item if original is outcome_missing else original
            for original, gathered_item in zip(worker_outcomes, gathered)
        ]
    except BaseException:
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        cancelled_settled: list[object] = []
        drain_task = aggregate
        try:
            while not drain_task.done():
                try:
                    await asyncio.shield(drain_task)
                except asyncio.CancelledError:
                    # A second caller cancellation must not strand workers or
                    # erase the usage they already consumed.
                    continue
            cancelled_settled = list(drain_task.result())
        finally:
            accounting_outcomes = [
                settled_item if original is outcome_missing else original
                for original, settled_item in zip(
                    worker_outcomes,
                    cancelled_settled,
                )
            ]
            _charge_failed_attempt_usage(
                pf.company,
                accounting_outcomes,
                token_budget=shared_token_budget,
            )
        raise
    attempt_usage_charged = False

    def charge_failed_attempt_usage() -> None:
        """Persist operational spend while leaving scientific state rolled back."""
        nonlocal attempt_usage_charged
        if attempt_usage_charged:
            return
        _charge_failed_attempt_usage(
            pf.company,
            list(settled),
            token_budget=shared_token_budget,
        )
        attempt_usage_charged = True

    failure = next(
        (item for item in settled if isinstance(item, BaseException)),
        None,
    )
    if failure is not None:
        charge_failed_attempt_usage()
        raise failure

    completed: list[tuple[Program, StageResult]] = []
    total_credits = 0.0
    total_tokens = 0
    for item in settled:
        program, result = item
        try:
            result = _validate_stage_result(program, result)
        except (TypeError, ValueError):
            charge_failed_attempt_usage()
            raise
        credits = float(result.credits_spent)
        tokens = int(result.tokens_spent)
        if (
            approved_allocation is not None
            and credits > approved_envelopes[program.id] + 1e-9
        ):
            charge_failed_attempt_usage()
            raise InsufficientCredits(
                f"stage result for {program.name!r} spent {float(credits):.1f} "
                f"credits, exceeding its approved CSO envelope "
                f"{approved_envelopes[program.id]:.1f}"
            )
        total_credits += credits
        total_tokens += tokens
        completed.append((program, result))

    exact_ledger = Ledger(working.company)
    if not exact_ledger.can_afford(total_credits):
        charge_failed_attempt_usage()
        raise InsufficientCredits(
            f"quarter results need {total_credits:.1f} credits; "
            f"only {exact_ledger.credits_remaining:.1f} remain"
        )
    if not exact_ledger.can_afford_tokens(total_tokens):
        charge_failed_attempt_usage()
        raise InsufficientTokens(
            f"quarter results need {total_tokens} tokens; "
            f"only {exact_ledger.tokens_remaining} remain"
        )

    committed: list[tuple[str, StageResult, GateRecommendation]] = []
    try:
        for program, result in completed:
            result, rec = _record_stage_result(working, program, result)
            committed.append((program.id, result, rec))
            if auto:
                resolve_gate(working, program, rec.decision)
    except BaseException:
        charge_failed_attempt_usage()
        raise

    _install_portfolio_state(pf, working)
    programs_by_id = {program.id: program for program in pf.programs}
    return [
        (programs_by_id[program_id], result, rec)
        for program_id, result, rec in committed
    ]


def _install_portfolio_state(target: Portfolio, source: Portfolio) -> None:
    """Install a completed transaction while preserving caller-held identities.

    The function contains no await points, so it is atomic with respect to other
    coroutines on the same event loop. Existing ``Company`` and ``Program``
    objects are updated in place because callers commonly retain references to
    them across quarters.
    """
    target.company.__dict__.clear()
    target.company.__dict__.update(source.company.__dict__)

    existing = {program.id: program for program in target.programs}
    installed: list[Program] = []
    for source_program in source.programs:
        program = existing.get(source_program.id)
        if program is None:
            program = source_program
        else:
            program.__dict__.clear()
            program.__dict__.update(source_program.__dict__)
        installed.append(program)

    target.programs[:] = installed
    target.pending.clear()
    target.pending.update(source.pending)
    target.gate_records[:] = source.gate_records
    target.experiments[:] = source.experiments
    target.stage_runs[:] = source.stage_runs
