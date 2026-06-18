"""The science layer — seeded, deterministic, CPU-only (v1 spine).

Everything here is reproducible given a seed: experiment readouts are draws from
a seeded RNG, so a run can be replayed exactly (spec §7). This is the *stub* that
stands in for live SOTA-model experiments and co-scientist runs; the readout
fidelity is therefore "T0" (LLM/heuristic-estimated) and labelled as such so the
gate packet never passes a simulated number off as a computed one (spec §5).

`run_stage` is the single seam where a real co-scientist run / Modal model call
gets swapped in later.
"""
from __future__ import annotations

import random
import zlib

from company.models import (
    BASELINE_POS,
    STAGE_CREDIT_COST,
    Candidate,
    Experiment,
    Program,
    Stage,
    StageResult,
)

# Generation methods that "vote" on a candidate (spec §4). Agreement across these
# is the repurposing confidence signal.
METHODS = ["kg", "signature", "target", "structure", "network", "literature"]


# --- Demo fixtures ----------------------------------------------------------
# Real PAH repurposing candidates with SYNTHETIC per-method scores. These are
# illustrative fixtures (is_fixture=True), NOT predictions — they exist so the org
# loop produces legible output. Sildenafil/tadalafil are the known-correct answers
# (approved for PAH), so the loop surfacing them via multi-method consensus is the
# spine's sanity check. Replaced by live KG/signature/literature methods later.
_PAH_FIXTURES: list[Candidate] = [
    Candidate("sildenafil", "pulmonary arterial hypertension",
              {"kg": 0.92, "signature": 0.81, "target": 0.95, "structure": 0.88,
               "network": 0.84, "literature": 0.97},
              rationale="PDE5 inhibition → cGMP-mediated pulmonary vasodilation."),
    Candidate("tadalafil", "pulmonary arterial hypertension",
              {"kg": 0.88, "signature": 0.74, "target": 0.93, "structure": 0.85,
               "network": 0.80, "literature": 0.90},
              rationale="Long-acting PDE5 inhibitor; same mechanism as sildenafil."),
    Candidate("imatinib", "pulmonary arterial hypertension",
              {"kg": 0.71, "signature": 0.66, "target": 0.78, "structure": 0.62,
               "network": 0.74, "literature": 0.69},
              rationale="PDGFR inhibition → anti-remodeling; efficacy offset by tolerability."),
    Candidate("metformin", "pulmonary arterial hypertension",
              {"kg": 0.52, "signature": 0.58, "target": 0.41, "structure": 0.0,
               "network": 0.55, "literature": 0.49},
              rationale="AMPK activation; preclinical anti-proliferative signal."),
    Candidate("spironolactone", "pulmonary arterial hypertension",
              {"kg": 0.44, "signature": 0.39, "target": 0.46, "structure": 0.0,
               "network": 0.42, "literature": 0.40},
              rationale="Mineralocorticoid antagonism; adjunctive interest only."),
    Candidate("loratadine", "pulmonary arterial hypertension",
              {"kg": 0.12, "signature": 0.18, "target": 0.08, "structure": 0.0,
               "network": 0.15, "literature": 0.10},
              rationale="Negative control — antihistamine with no PAH rationale."),
]


def candidate_fixtures(disease: str) -> list[Candidate]:
    """Return seeded demo candidates for a disease (PAH only in v1)."""
    if "pulmonary arterial hypertension" in disease.lower() or "pah" in disease.lower():
        return [Candidate(c.drug, c.indication, dict(c.method_scores), c.rationale) for c in _PAH_FIXTURES]
    return []


# --- Scoring & integration --------------------------------------------------

def integrate_candidate(c: Candidate) -> tuple[float, float]:
    """Fuse a candidate's per-method scores into (confidence, method_agreement).

    Confidence is the mean of non-zero method scores; agreement is high when the
    contributing methods cluster tightly (low spread) and many methods contribute.
    More agreeing methods → tighter experiment variance downstream (spec §7).
    """
    vals = [v for v in c.method_scores.values() if v > 0]
    if not vals:
        return 0.0, 0.0
    confidence = sum(vals) / len(vals)
    spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
    coverage = len(vals) / len(METHODS)
    agreement = max(0.0, min(1.0, (1.0 - spread) * (0.5 + 0.5 * coverage)))
    return confidence, agreement


def _stable_hash(s: str) -> int:
    """Process-stable hash. Builtin hash() randomizes strings per-process (PYTHONHASHSEED),
    which would break the spec's seeded-reproducibility promise across runs."""
    return zlib.crc32(s.encode())


def _seed_for(program: Program, stage: Stage, cycle: int) -> int:
    return (program.seed * 1_000_003 + _stable_hash(stage.value) % 100_003 + cycle * 31) & 0x7FFFFFFF


def run_experiment(rng: random.Random, *, program_id: str, stage: Stage, assay: str,
                   method: str, confidence: float, agreement: float, credits: float) -> Experiment:
    """A single seeded in-silico readout ~ N(confidence, sigma), tightened by agreement."""
    sigma = 0.28 * (1.0 - 0.6 * agreement)
    readout = max(0.0, min(1.0, rng.gauss(confidence, sigma)))
    interp = ("strong" if readout >= 0.66 else "weak" if readout < 0.4 else "moderate")
    return Experiment(
        program_id=program_id, stage=stage.value, assay=assay, method=method,
        fidelity_tier="T0", readout=round(readout, 3),
        interpretation=f"{interp} signal ({readout:.2f})", seed=rng.randint(0, 2**31 - 1),
        credits=round(credits, 2),
    )


def run_stage(program: Program, cycle: int) -> StageResult:
    """Run one stage for a program (v1 seeded stub). Returns a StageResult.

    THE INTEGRATION SEAM: replace the body with a real co-scientist run + Modal
    model calls. The return contract (confidence, agreement, red_flags, candidates,
    experiments, costs) is what the gate/ledger consume.
    """
    stage = program.stage
    rng = random.Random(_seed_for(program, stage, cycle))
    credit_cost = STAGE_CREDIT_COST[stage]

    candidates: list[Candidate] = []
    experiments: list[Experiment] = []

    if stage is Stage.HYPOTHESES:
        from company import kg
        cands = candidate_fixtures(program.disease)
        # LIVE METHOD: replace the synthetic `network` vote with a real KG-on-CPU
        # network-proximity score (T1) computed per candidate (build-path step 1).
        kg.annotate_network_scores(cands)
        scored = []
        for c in cands:
            conf, agr = integrate_candidate(c)
            scored.append((conf, agr, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:5]
        candidates = [c for _, _, c in top]
        # The program's confidence is led by its best consensus candidate.
        confidence, agreement, lead = top[0]
        program.lead_candidate = f"{lead.drug} -> {lead.indication}"
        for conf, agr, c in top:
            experiments.append(run_experiment(
                rng, program_id=program.id, stage=stage, assay="consensus_screen",
                method="multi_method", confidence=conf, agreement=agr,
                credits=credit_cost / max(1, len(top)),
            ))
        red_flags = 0
    else:
        # Generic stages: seeded confidence anchored on the program's running
        # confidence (so evidence compounds), plus a couple of assay readouts.
        anchor = program.confidence or rng.uniform(0.5, 0.75)
        confidence = max(0.0, min(1.0, rng.gauss(anchor, 0.12)))
        agreement = rng.uniform(0.5, 0.85)
        assays = {
            Stage.DISEASE_DOSSIER: ["target_enrichment", "pathway_coherence"],
            Stage.MECHANISM: ["docking_affinity", "network_proximity"],
            Stage.TRANSLATIONAL: ["rwe_signal", "feasibility"],
        }[stage]
        for a in assays:
            experiments.append(run_experiment(
                rng, program_id=program.id, stage=stage, assay=a, method=a,
                confidence=confidence, agreement=agreement,
                credits=credit_cost / len(assays),
            ))
        # Mechanism stage can surface liabilities (the realistic risk locus).
        red_flags = sum(1 for e in experiments if e.readout < 0.4)

    return StageResult(
        stage=stage.value, confidence=round(confidence, 3),
        method_agreement=round(agreement, 3), red_flags=red_flags,
        top_candidates=candidates, experiments=experiments,
        credits_spent=credit_cost, tokens_spent=rng.randint(8000, 30000),
    )


# --- Risk & value -----------------------------------------------------------

def realized_pos(stage: Stage, confidence: float, red_flags: int) -> float:
    """Baseline transition PoS modulated by the science (spec §7).

    Strong confidence nudges PoS up; each red flag drags it down. Clamped to a
    sane band so a single stage can neither guarantee nor doom a program.
    """
    base = BASELINE_POS[stage]
    modulated = base * (0.6 + 0.8 * confidence) - 0.07 * red_flags
    return max(0.05, min(0.95, modulated))


def rnpv_contribution(program: Program) -> float:
    """Risk-adjusted value minus spend, in abstract credit-equivalent units (spec §7).

    estimated_value and credits_spent share units so the difference is meaningful.
    """
    return program.cumulative_pos * program.estimated_value - program.credits_spent
