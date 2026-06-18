"""The CSO — portfolio strategy (v1 deterministic allocator).

The CSO *proposes*; the CEO disposes (spec §8). v1 is a transparent rule-based
allocator (no LLM call needed, fully reproducible). The seam to upgrade it to an
LLM agent reasoning over portfolio state is `recommend_gate` / `propose_allocation`.
"""
from __future__ import annotations

from dataclasses import dataclass

from company.models import GateDecision, Program, Stage, StageResult, next_stage
from company.models import STAGE_CREDIT_COST
from company.science import realized_pos, rnpv_contribution

KILL_CONFIDENCE = 0.45      # below this, recommend killing
HOLD_FLAG_THRESHOLD = 2     # this many red flags with middling confidence → hold


@dataclass
class GateRecommendation:
    decision: GateDecision
    rationale: str
    confidence: float
    realized_pos: float
    rnpv_contribution: float


def recommend_gate(program: Program, result: StageResult) -> GateRecommendation:
    """Recommend advance/kill/hold for a program that just finished a stage."""
    pos = realized_pos(program.stage, result.confidence, result.red_flags)
    rnpv = rnpv_contribution(program)

    if result.confidence < KILL_CONFIDENCE:
        decision = GateDecision.KILL
        why = (f"confidence {result.confidence:.2f} < {KILL_CONFIDENCE} kill line; "
               f"rNPV {rnpv:+.0f} — free the credits for stronger programs")
    elif result.red_flags >= HOLD_FLAG_THRESHOLD and result.confidence < 0.62:
        decision = GateDecision.HOLD
        why = (f"{result.red_flags} red flags at confidence {result.confidence:.2f} — "
               f"run one more cycle to de-risk before committing GPU spend")
    else:
        decision = GateDecision.ADVANCE
        nxt = next_stage(program.stage)
        why = (f"confidence {result.confidence:.2f}, PoS {pos:.2f}, rNPV {rnpv:+.0f} — "
               f"advance to {nxt.value}")
    return GateRecommendation(decision, why, result.confidence, pos, rnpv)


def propose_allocation(programs: list[Program], credits_remaining: float) -> dict[str, float]:
    """Split remaining credits across active programs ~ expected marginal rNPV per credit.

    Higher risk-adjusted value and cheaper next stage → more credits. Programs that
    can't afford their next stage get zero (the CEO must kill or refill).
    """
    active = [p for p in programs if p.status.value == "active" and p.stage is not Stage.COMPLETED]
    weights: dict[str, float] = {}
    for p in active:
        next_cost = STAGE_CREDIT_COST.get(p.stage, 1.0)
        # marginal value ≈ risk-adjusted value gated by how cheaply we can learn more
        weights[p.id] = max(0.0, p.cumulative_pos * p.estimated_value) / max(1.0, next_cost)
    total = sum(weights.values())
    if total <= 0:
        return {p.id: 0.0 for p in active}
    return {pid: round(credits_remaining * w / total, 2) for pid, w in weights.items()}
