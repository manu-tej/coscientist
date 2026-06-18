"""The company engine — program/gate state machine with stochastic attrition.

A `Portfolio` holds the company, its programs, pending gate packets, and the audit
trail (experiments + gate records). The engine advances simulated quarters: run
each active program's current stage (debiting the ledger once), produce a gate
packet, and apply the CEO's decision. On ADVANCE a seeded dice-roll against the
realized PoS can still kill the program — stochastic, data-realized attrition
(spec §7, §9).
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field

from company.cso import GateRecommendation, recommend_gate
from company.ledger import Ledger
from company.models import (
    Company,
    GateDecision,
    GateRecord,
    Program,
    ProgramStatus,
    Stage,
    StageResult,
    next_stage,
)
from company.science import realized_pos, rnpv_contribution, run_stage


@dataclass
class Portfolio:
    company: Company
    programs: list[Program] = field(default_factory=list)
    pending: dict[str, StageResult] = field(default_factory=dict)   # program_id -> result awaiting a gate
    gate_records: list[GateRecord] = field(default_factory=list)
    experiments: list = field(default_factory=list)                 # full Experiment audit trail

    def program(self, pid: str) -> Program:
        for p in self.programs:
            if p.id == pid or p.name == pid:
                return p
        raise KeyError(f"no program {pid!r}")

    def active_programs(self) -> list[Program]:
        return [p for p in self.programs if p.status is ProgramStatus.ACTIVE]


def new_company(name: str, disease: str, *, token_budget: int = 5_000_000,
                credit_budget: float = 500.0, seed: int = 0) -> Portfolio:
    company = Company(id=str(uuid.uuid4())[:8], name=name, disease_focus=disease,
                      token_budget=token_budget, credit_budget=credit_budget, seed=seed)
    return Portfolio(company=company)


def add_program(pf: Portfolio, name: str, disease: str, *, estimated_value: float = 400.0,
                seed: int | None = None) -> Program:
    if seed is None:
        seed = pf.company.seed + len(pf.programs) + 1
    p = Program(id=str(uuid.uuid4())[:8], company_id=pf.company.id, name=name,
                disease=disease, estimated_value=estimated_value, seed=seed)
    p.history.append(f"founded @ cycle {pf.company.cycle}; est. value {estimated_value:.0f}")
    pf.programs.append(p)
    return p


def run_program_stage(pf: Portfolio, program: Program) -> tuple[StageResult, GateRecommendation]:
    """Run the program's current stage once, debit the ledger, stash a pending gate."""
    ledger = Ledger(pf.company)
    result = run_stage(program, pf.company.cycle)
    ledger.debit(program, credits=result.credits_spent, tokens=result.tokens_spent)
    program.confidence = result.confidence
    program.red_flags = result.red_flags
    pf.experiments.extend(result.experiments)
    pf.pending[program.id] = result
    rec = recommend_gate(program, result)
    program.history.append(
        f"cycle {pf.company.cycle}: ran {program.stage.value} → conf {result.confidence:.2f}, "
        f"{result.red_flags} flags, spent {result.credits_spent:.1f}cr; CSO: {rec.decision.value}"
    )
    return result, rec


def _gate_rng(program: Program, cycle: int) -> random.Random:
    import zlib
    stable = zlib.crc32(program.stage.value.encode())
    return random.Random((program.seed * 7919 + cycle * 104_729 + stable) & 0x7FFFFFFF)


def resolve_gate(pf: Portfolio, program: Program, decision: GateDecision) -> GateRecord:
    """Apply a CEO gate decision to a program with a pending stage result."""
    result = pf.pending.pop(program.id, None)
    if result is None:
        raise ValueError(f"program {program.name!r} has no pending gate; run a quarter first")

    pos = realized_pos(program.stage, result.confidence, result.red_flags)
    transition = f"{program.stage.value}->{next_stage(program.stage).value}"
    rec = recommend_gate(program, result)
    record = GateRecord(
        program_id=program.id, transition=transition, confidence=result.confidence,
        realized_pos=round(pos, 3), red_flags=result.red_flags,
        rnpv_contribution=round(rnpv_contribution(program), 1),
        cso_recommendation=rec.decision.value, ceo_decision=decision.value,
    )

    if decision is GateDecision.KILL:
        program.status = ProgramStatus.KILLED
        record.note = "CEO killed the program"
        program.history.append(f"cycle {pf.company.cycle}: GATE killed by CEO")

    elif decision is GateDecision.HOLD:
        record.note = "held — will re-run the stage next quarter"
        program.history.append(f"cycle {pf.company.cycle}: GATE held by CEO")

    else:  # ADVANCE — stochastic attrition can still kill it
        survived = _gate_rng(program, pf.company.cycle).random() < pos
        record.survived_roll = survived
        program.cumulative_pos = round(program.cumulative_pos * pos, 4)
        if not survived:
            program.status = ProgramStatus.KILLED
            record.note = f"advanced but FAILED stochastically in {program.stage.value} (PoS {pos:.2f})"
            program.history.append(
                f"cycle {pf.company.cycle}: GATE advanced → FAILED roll (PoS {pos:.2f}) — attrition")
        else:
            nxt = next_stage(program.stage)
            program.stage = nxt
            if nxt is Stage.COMPLETED:
                program.status = ProgramStatus.COMPLETED
                record.note = "cleared final gate → clinical-hypothesis package"
                program.history.append(f"cycle {pf.company.cycle}: COMPLETED — {program.lead_candidate}")
            else:
                record.note = f"advanced to {nxt.value} (survived PoS {pos:.2f})"
                program.history.append(f"cycle {pf.company.cycle}: GATE advanced → {nxt.value}")

    pf.gate_records.append(record)
    return record


def run_quarter(pf: Portfolio, *, auto: bool = False) -> list[tuple[Program, StageResult, GateRecommendation]]:
    """Advance one simulated quarter: run each active program's current stage.

    With auto=True, immediately resolve each gate by the CSO's recommendation (used
    for the reproducible demo + tests). Otherwise leaves pending gates for the CEO.
    """
    pf.company.cycle += 1
    out: list[tuple[Program, StageResult, GateRecommendation]] = []
    for program in pf.active_programs():
        if program.id in pf.pending:
            continue  # already awaiting a CEO decision
        result, rec = run_program_stage(pf, program)
        out.append((program, result, rec))
        if auto:
            resolve_gate(pf, program, rec.decision)
    return out
