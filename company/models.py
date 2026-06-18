"""Data model + stage configuration for the repurposing techbio.

Enums and dataclasses mirror the conventions in `core/models.py`. Stage
constants (order, baseline PoS, costs) live here too since they are pure data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Stage(Enum):
    """The repurposing value chain. A program advances one stage per approved gate."""
    DISEASE_DOSSIER = "disease_dossier"   # characterize disease mechanism/targets/signature
    HYPOTHESES = "hypotheses"             # propose (drug -> indication) candidates
    MECHANISM = "mechanism"               # mechanistic validation (binding/network/pathway)
    TRANSLATIONAL = "translational"       # RWE, prior trials, feasibility, safety reuse
    COMPLETED = "completed"               # cleared the final gate


# Ordered pipeline; COMPLETED is terminal and not "run".
STAGE_ORDER: list[Stage] = [
    Stage.DISEASE_DOSSIER,
    Stage.HYPOTHESES,
    Stage.MECHANISM,
    Stage.TRANSLATIONAL,
]

# Baseline probability of *successfully transitioning out of* each stage. Elevated
# vs de-novo discovery because an approved drug's safety/PK is already known, so
# attrition concentrates in efficacy/mechanism. Illustrative — tune during build (spec §7).
BASELINE_POS: dict[Stage, float] = {
    Stage.DISEASE_DOSSIER: 0.65,
    Stage.HYPOTHESES: 0.55,
    Stage.MECHANISM: 0.55,
    Stage.TRANSLATIONAL: 0.50,
}

# Modal GPU credits a stage run nominally costs (the cheap CPU/literature stages
# cost little; mechanism does the GPU docking). Token cost is tracked separately.
STAGE_CREDIT_COST: dict[Stage, float] = {
    Stage.DISEASE_DOSSIER: 2.0,
    Stage.HYPOTHESES: 5.0,
    Stage.MECHANISM: 25.0,    # Boltz-2 / DiffDock on Modal — the expensive one
    Stage.TRANSLATIONAL: 8.0,
}


def next_stage(stage: Stage) -> Stage:
    """Stage that follows `stage`; COMPLETED once the last stage is cleared."""
    i = STAGE_ORDER.index(stage)
    return STAGE_ORDER[i + 1] if i + 1 < len(STAGE_ORDER) else Stage.COMPLETED


class ProgramStatus(Enum):
    ACTIVE = "active"        # in the pipeline, between or within stages
    HELD = "held"            # CEO paused it; no credits committed
    KILLED = "killed"        # terminated (CEO decision or stochastic failure)
    COMPLETED = "completed"  # cleared the final gate — a clinical-hypothesis package


class GateDecision(Enum):
    ADVANCE = "advance"
    KILL = "kill"
    HOLD = "hold"


@dataclass
class Candidate:
    """A (drug -> indication) repurposing hypothesis with per-method evidence.

    `method_scores` keys are generation methods (kg, signature, target, structure,
    network, literature); values are in [0, 1]. The spread/agreement across methods
    drives confidence (consensus) and experiment variance (spec §4, §7).
    """
    drug: str
    indication: str
    method_scores: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    elo_rating: float = 1200.0
    is_fixture: bool = True   # v1: seeded demo fixture, not a real prediction

    @property
    def n_methods(self) -> int:
        return sum(1 for v in self.method_scores.values() if v > 0)

    @property
    def mean_score(self) -> float:
        vals = [v for v in self.method_scores.values() if v > 0]
        return sum(vals) / len(vals) if vals else 0.0


@dataclass
class Experiment:
    """One in-silico assay readout. Records provenance for auditability (spec §10)."""
    program_id: str
    stage: str
    assay: str
    method: str
    fidelity_tier: str        # "T0" | "T1" | "T2"
    readout: float            # in [0, 1]
    interpretation: str
    seed: int
    credits: float


@dataclass
class GateRecord:
    """A gate decision and its (possibly stochastic) outcome — the audit trail."""
    program_id: str
    transition: str           # e.g. "hypotheses->mechanism"
    confidence: float
    realized_pos: float
    red_flags: int
    rnpv_contribution: float
    cso_recommendation: str
    ceo_decision: Optional[str] = None
    survived_roll: Optional[bool] = None   # stochastic attrition outcome (spec §7)
    note: str = ""


@dataclass
class StageResult:
    """Output of running one stage for a program (the stub or, later, a co-scientist run)."""
    stage: str
    confidence: float                 # integrated confidence in [0, 1]
    method_agreement: float           # in [0, 1]; tightens experiment variance
    red_flags: int
    top_candidates: list[Candidate] = field(default_factory=list)
    experiments: list[Experiment] = field(default_factory=list)
    credits_spent: float = 0.0
    tokens_spent: int = 0


@dataclass
class Program:
    id: str
    company_id: str
    name: str
    disease: str
    stage: Stage = Stage.DISEASE_DOSSIER
    status: ProgramStatus = ProgramStatus.ACTIVE
    seed: int = 0
    estimated_value: float = 0.0      # peak commercial value ($M-ish, abstract units)
    cumulative_pos: float = 1.0       # product of realized transition PoS so far
    confidence: float = 0.0           # latest stage integrated confidence
    red_flags: int = 0
    credits_spent: float = 0.0
    tokens_spent: int = 0
    lead_candidate: Optional[str] = None   # "drug -> indication" once known
    history: list[str] = field(default_factory=list)  # human-readable event log

    @property
    def is_active(self) -> bool:
        return self.status in (ProgramStatus.ACTIVE, ProgramStatus.HELD)


@dataclass
class Company:
    id: str
    name: str
    disease_focus: str
    token_budget: int
    credit_budget: float          # total Modal GPU credits (e.g. 500)
    token_spent: int = 0
    credit_spent: float = 0.0
    cycle: int = 0                # simulated "quarter"
    seed: int = 0
