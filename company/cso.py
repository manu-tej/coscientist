"""The CSO — portfolio strategy (v1 deterministic allocator).

The CSO *proposes*; the CEO disposes (spec §8). v1 is a transparent rule-based
allocator (no LLM call needed, fully reproducible). The seam to upgrade it to an
LLM agent reasoning over portfolio state is `recommend_gate` / `propose_allocation`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from company.models import (
    GateDecision,
    Program,
    ProgramStatus,
    Stage,
    StageResult,
    next_stage,
)
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


def marginal_rnpv_per_credit(program: Program) -> float:
    """Expected risk-adjusted value unlocked by clearing the NEXT gate, per credit.

    `cumulative_pos × PoS(next gate) × estimated_value / next_stage_cost` — so a
    program earns credits for being valuable, likely to clear its gate (which the
    science modulates via confidence and red flags), and cheap to learn more about.
    This is the information-value signal §8 asks for, not a bare value ranking.
    """
    if program.stage is Stage.COMPLETED:
        return 0.0
    next_cost = max(1.0, STAGE_CREDIT_COST.get(program.stage, 1.0))
    pos = realized_pos(program.stage, program.confidence, program.red_flags)
    return max(0.0, program.cumulative_pos * pos * program.estimated_value) / next_cost


def explore_exploit_beta(cycle: int) -> float:
    """Concentration exponent on the allocation weights: spread early, concentrate late.

    beta≈1 early (explore — fund many programs to buy information), rising to ~4
    (exploit — pile credits onto validated leaders). The bandit-flavored curve §8 wants.
    """
    return min(4.0, 1.0 + 0.5 * max(0, cycle))


def propose_allocation(programs: list[Program], credits_remaining: float,
                       *, cycle: int = 0,
                       pending_ids: set[str] | None = None) -> dict[str, float]:
    """Fund launch-eligible ACTIVE/HELD programs by marginal rNPV per credit.

    HELD programs need an allocation to run their de-risking cycle. Programs
    already awaiting a CEO gate are excluded so their share is not stranded.
    """
    pending_ids = pending_ids or set()
    active = [
        p for p in programs
        if p.status in (ProgramStatus.ACTIVE, ProgramStatus.HELD)
        and p.stage is not Stage.COMPLETED
        and p.id not in pending_ids
    ]
    beta = explore_exploit_beta(cycle)
    weights = {p.id: marginal_rnpv_per_credit(p) ** beta for p in active}
    total = sum(weights.values())
    if total <= 0:
        return {p.id: 0.0 for p in active}
    return {pid: round(credits_remaining * w / total, 2) for pid, w in weights.items()}


# --- portfolio planning (multi-program loop, spec §8) ------------------------

# Diseases the CSO may propose founding programs against, with rough peak values.
CANDIDATE_DISEASES: dict[str, float] = {
    "pulmonary arterial hypertension": 400.0,
    "multiple myeloma": 900.0,
    "breast cancer": 1200.0,
    "type 2 diabetes": 1100.0,
    "idiopathic pulmonary fibrosis": 700.0,
    "amyotrophic lateral sclerosis": 500.0,
    "androgenetic alopecia": 300.0,
}


@dataclass
class PortfolioAction:
    kind: str          # "gate" | "fund" | "found"
    target: str        # program name, or disease for "found"
    detail: str        # the concrete ask (decision, credit amount, est. value)
    rationale: str
    priority: float    # ranking score — higher acts first


@dataclass
class PortfolioPlan:
    actions: list[PortfolioAction]
    allocation: dict[str, float]
    source: str = "deterministic"   # "deterministic" | "llm"
    summary: str = ""

    def targets(self) -> list[str]:
        return [a.target for a in self.actions]


def propose_new_programs(programs: list[Program], k: int = 2) -> list[tuple[str, float]]:
    """Highest-value candidate diseases with no live program yet (spec §8.3)."""
    covered = {p.disease.strip().lower() for p in programs
               if p.status.value in ("active", "held", "completed")}
    open_ = [(d, v) for d, v in CANDIDATE_DISEASES.items() if d.strip().lower() not in covered]
    return sorted(open_, key=lambda t: t[1], reverse=True)[:k]


def deterministic_plan(programs: list[Program], pending: dict, credits_remaining: float,
                       *, cycle: int = 0, n_new: int = 2) -> PortfolioPlan:
    """The quantitative plan: gate calls, credit allocation, and new-program proposals,
    ranked into one action list. Fully reproducible — no LLM involved."""
    alloc = propose_allocation(
        programs,
        credits_remaining,
        cycle=cycle,
        pending_ids=set(pending),
    )
    by_id = {p.id: p for p in programs}
    actions: list[PortfolioAction] = []

    for pid, result in pending.items():
        p = by_id.get(pid)
        if p is None:
            continue
        rec = recommend_gate(p, result)
        actions.append(PortfolioAction(
            "gate", p.name, rec.decision.value.upper(), rec.rationale,
            priority=1000.0 + abs(rec.rnpv_contribution)))   # gates are always most urgent

    for pid, credits in alloc.items():
        p = by_id.get(pid)
        if p is None or credits <= 0:
            continue
        mv = marginal_rnpv_per_credit(p)
        actions.append(PortfolioAction(
            "fund", p.name, f"{credits:.1f} credits",
            f"marginal rNPV/credit {mv:.1f} at {p.stage.value} "
            f"(cumPoS {p.cumulative_pos:.2f}, value {p.estimated_value:.0f})",
            priority=mv))

    for disease, value in propose_new_programs(programs, n_new):
        actions.append(PortfolioAction(
            "found", disease, f"est. value {value:.0f}",
            "high-value indication with no live program — found to widen the portfolio",
            priority=value / 100.0))

    actions.sort(key=lambda a: a.priority, reverse=True)
    return PortfolioPlan(actions=actions, allocation=alloc, source="deterministic",
                         summary=f"{len(actions)} actions at cycle {cycle} "
                                 f"(beta={explore_exploit_beta(cycle):.1f})")


def _parse_order(text: str, valid: list[str]) -> list[str]:
    """Pull an ordered list of action targets from an LLM reply, tolerant of prose
    or JSON. Order-preserving, de-duplicated, validated against `valid`."""
    picked: list[str] = []
    m = re.search(r"\[[^\]]*\]", text, re.DOTALL)
    if m:
        try:
            for item in json.loads(m.group(0)):
                name = str(item).strip()
                match = next((v for v in valid if v.lower() == name.lower()), None)
                if match and match not in picked:
                    picked.append(match)
        except (json.JSONDecodeError, TypeError):
            pass
    if not picked:   # fall back to mention order in the prose
        # Word-boundary matched: a short program name like "A" must not match the
        # "a" inside an unrelated word, or any prose reply would parse as a ranking.
        hits = []
        for v in valid:
            m2 = re.search(rf"\b{re.escape(v)}\b", text, re.IGNORECASE)
            if m2:
                hits.append((m2.start(), v))
        picked = [v for _, v in sorted(hits)]
    return picked


class CSOAgent:
    """The portfolio brain. The *numbers* stay computed (allocation, PoS, rNPV);
    the LLM does what arithmetic can't — weigh strategy and re-prioritize the action
    list, with a written rationale. Without a backend (or on any failure) it degrades
    to the deterministic plan, so the loop always runs offline and reproducibly.
    """

    def __init__(self, backend=None):
        self.backend = backend

    async def propose_plan(self, programs: list[Program], pending: dict,
                           credits_remaining: float, *, cycle: int = 0) -> PortfolioPlan:
        base = deterministic_plan(programs, pending, credits_remaining, cycle=cycle)
        if self.backend is None or not base.actions:
            return base

        listing = "\n".join(
            f"- {a.target} [{a.kind}] {a.detail} — {a.rationale}" for a in base.actions)
        system = ("You are the CSO of an AI-native drug-repurposing company. You weigh "
                  "portfolio strategy: explore vs exploit, when to cut losses, when to widen "
                  "the portfolio. You do NOT invent numbers — the quantitative scores are given.")
        user = (f"Cycle {cycle}. Remaining budget: {credits_remaining:.0f} Modal credits.\n\n"
                f"Proposed actions (quantitatively ranked):\n{listing}\n\n"
                f"Re-rank these actions by strategic priority for the CEO. Reply with a JSON "
                f"array of the action targets in your preferred order, then 2-3 sentences "
                f"explaining the strategy.")
        try:
            reply = await self.backend.call(system, user, max_tokens=800)
        except Exception as exc:   # rate limit / backend error — degrade, never crash the loop
            base.summary = f"LLM error ({exc}); deterministic plan"
            return base

        order = _parse_order(reply, base.targets())
        if not order:
            base.summary = "unparseable LLM reply; deterministic plan"
            return base
        rank = {t: i for i, t in enumerate(order)}
        reordered = sorted(base.actions, key=lambda a: (rank.get(a.target, len(order)), -a.priority))
        return PortfolioPlan(actions=reordered, allocation=base.allocation,
                             source="llm", summary=reply.strip()[:400])
