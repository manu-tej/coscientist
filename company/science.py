"""The quantitative science layer and reproducible T0 baseline.

Everything here is reproducible given a seed: experiment readouts are draws from
a seeded RNG, so a run can be replayed exactly (spec §7). Seeded readouts are
labelled T0 so a gate packet never passes a simulation off as computed evidence.
The registry can replace individual facets with live T1 CPU/network or T2 GPU
providers. ``company.hierarchy.CoScientistStageExecutor`` supplies this baseline
to a real multi-agent stage team and validates its final synthesis.
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
from company.evidence import build_candidate_evidence
from company.registry import build_default_registry

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


# Approved-drug library screened against any disease that has no curated fixtures.
# Real, approved drugs — their scores come entirely from the live methods (registry),
# never from hand-typed values, so this generalizes the pipeline off PAH.
_DRUG_POOL: list[str] = [
    "sildenafil", "tadalafil", "imatinib", "metformin", "spironolactone", "loratadine",
    "thalidomide", "lenalidomide", "minoxidil", "finasteride", "raloxifene", "tamoxifen",
]


def regulatory_record(drug: str) -> dict[str, object]:
    """Return the local regulatory evidence available for a candidate name.

    This curated screen is not a live FDA/EMA query.  Unknown names remain
    explicitly unknown so downstream packages never infer approval from an
    agent-generated mention.
    """
    normalized = drug.strip().casefold()
    if normalized in {name.casefold() for name in _DRUG_POOL}:
        return {
            "status": "curated_approved",
            "source": "company.science approved-drug screen v1",
            "live_verified": False,
        }
    return {
        "status": "unknown",
        "source": "not present in the local approved-drug screen",
        "live_verified": False,
    }


def candidate_pool(disease: str) -> list[Candidate]:
    """Candidates to screen for a disease: curated fixtures where we have them,
    otherwise the approved-drug library with NO prior scores — every vote must then
    come from a live method (KG / literature), or the candidate simply abstains."""
    fixtures = candidate_fixtures(disease)
    if fixtures:
        return fixtures
    return [Candidate(drug, disease, {}, is_fixture=False,
                      rationale="approved-drug library screen (no curated fixture)")
            for drug in _DRUG_POOL]


# --- Scoring & integration --------------------------------------------------

def integrate_candidate(c: Candidate) -> tuple[float, float]:
    """Fuse a candidate's per-method scores into (confidence, method_agreement).

    Confidence is the mean positive signal multiplied by coverage across the
    declared independent methods. Thus one 0.9 vote plus five abstentions is 0.15,
    not 0.9. Agreement is high when contributing methods cluster tightly and many
    methods contribute. More agreeing methods tighten experiment variance.
    """
    vals = [v for v in c.method_scores.values() if v > 0]
    if not vals:
        return 0.0, 0.0
    coverage = min(1.0, len(vals) / len(METHODS))
    confidence = (sum(vals) / len(vals)) * coverage
    spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
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
    experiment_seed = rng.randint(0, 2**31 - 1)
    experiment_rng = random.Random(experiment_seed)
    sigma = 0.28 * (1.0 - 0.6 * agreement)
    readout = max(0.0, min(1.0, experiment_rng.gauss(confidence, sigma)))
    interp = ("strong" if readout >= 0.66 else "weak" if readout < 0.4 else "moderate")
    return Experiment(
        program_id=program_id, stage=stage.value, assay=assay, method=method,
        fidelity_tier="T0", readout=round(readout, 3),
        interpretation=f"{interp} signal ({readout:.2f})", seed=experiment_seed,
        credits=round(credits, 2),
    )


# method key (on a Candidate) -> registry facet that resolves its vote
_LIVE_FACETS = {"network": "network_proximity", "literature": "literature",
                "structure": "structure"}


def estimate_stage_credits(
    program: Program,
    *,
    registry=None,
    allow_expensive: bool = True,
) -> float:
    """Conservative credit reservation before any provider call."""
    reg = registry or build_default_registry()
    base = STAGE_CREDIT_COST[program.stage]
    provider = 0.0
    if program.stage is Stage.HYPOTHESES:
        provider = len(candidate_pool(program.disease)) * sum(
            reg.max_provider_credits(
                facet, allow_expensive=allow_expensive,
            )
            for facet in _LIVE_FACETS.values()
        )
    elif program.stage is Stage.MECHANISM:
        provider = reg.max_provider_credits(
            "affinity", allow_expensive=allow_expensive,
        )
    return round(base + provider, 2)


def run_stage(program: Program, cycle: int, *, registry=None,
              allow_expensive: bool = True) -> StageResult:
    """Compute a stage's quantitative baseline and return its stable contract.

    The return contract (confidence, agreement, red flags, candidates,
    experiments, and costs) is consumed by both the seeded and hierarchical
    executors. Per-method scores resolve through the model ``registry`` (default:
    ``build_default_registry()``), which records the fidelity tier that answered
    each facet. ``allow_expensive`` gates the cost-tiered cascade so GPU providers
    run only for portfolio leaders.
    """
    stage = program.stage
    rng = random.Random(_seed_for(program, stage, cycle))
    credit_cost = STAGE_CREDIT_COST[stage]
    reg = registry or build_default_registry()

    candidates: list[Candidate] = []
    experiments: list[Experiment] = []
    provenance: dict[str, int] = {}
    provider_credits = 0.0

    def _bump(tier: str) -> None:
        provenance[tier] = provenance.get(tier, 0) + 1

    if stage is Stage.HYPOTHESES:
        cands = candidate_pool(program.disease)
        # Each live per-method vote is resolved through the registry: a real method
        # (KG proximity / PubMed co-occurrence, T1) where it can speak, the fixture
        # estimate (T0) as the floor. Independent methods → real consensus signal.
        for c in cands:
            votes = []
            for method_key, facet in _LIVE_FACETS.items():
                out = reg.resolve(
                    facet,
                    {
                        "drug": c.drug, "indication": c.indication,
                        "fixture_score": c.method_scores.get(method_key, 0.0),
                    },
                    allow_expensive=allow_expensive,
                )
                c.method_scores[method_key] = out.value
                votes.append((method_key, out))
                _bump(out.tier.value)
                provider_credits += out.credits
            # Provenance is first-class: every candidate keeps the chain from its
            # score back to the method and the concrete evidence behind it (§10).
            c.evidence = build_candidate_evidence(c.drug, c.indication, votes)
        scored = []
        for c in cands:
            conf, agr = integrate_candidate(c)
            scored.append((conf, agr, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        # Only candidates some method could actually speak to. If every method
        # abstained (no curated fixtures and no live method covers this disease),
        # the stage honestly returns zero confidence rather than inventing one.
        top = [t for t in scored if t[0] > 0][:5]
        candidates = [c for _, _, c in top]
        if not top:
            program.lead_candidate = None
            confidence, agreement, red_flags = 0.0, 0.0, 1
        else:
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
            center = confidence
            # MECHANISM docking_affinity routes through the registry's affinity facet
            # so the cost-tiered cascade applies: a Modal GPU provider runs only when
            # this program leads (allow_expensive); else the T0 estimate floor answers.
            if stage is Stage.MECHANISM and a == "docking_affinity":
                out = reg.resolve(
                    "affinity",
                    {
                        "drug": program.lead_drug or "",
                        "indication": program.lead_indication or program.disease,
                        "prior": confidence,
                    },
                    allow_expensive=allow_expensive,
                )
                center = out.value
                _bump(out.tier.value)
                provider_credits += out.credits
                interp = (
                    "strong" if center >= 0.66
                    else "weak" if center < 0.4
                    else "moderate"
                )
                experiments.append(Experiment(
                    program_id=program.id,
                    stage=stage.value,
                    assay=a,
                    method=out.provider,
                    fidelity_tier=out.tier.value,
                    readout=round(center, 3),
                    interpretation=f"{interp} signal ({center:.2f})",
                    seed=_seed_for(program, stage, cycle),
                    credits=round(out.credits, 2),
                    provider=out.provider,
                    detail=out.detail,
                    evidence=dict(out.evidence),
                ))
                continue
            experiments.append(run_experiment(
                rng, program_id=program.id, stage=stage, assay=a, method=a,
                confidence=center, agreement=agreement,
                credits=credit_cost / len(assays),
            ))
        # Mechanism stage can surface liabilities (the realistic risk locus).
        red_flags = sum(1 for e in experiments if e.readout < 0.4)

    return StageResult(
        stage=stage.value, confidence=round(confidence, 3),
        method_agreement=round(agreement, 3), red_flags=red_flags,
        top_candidates=candidates, experiments=experiments,
        credits_spent=round(credit_cost + provider_credits, 2),
        tokens_spent=rng.randint(8000, 30000),
        method_provenance=provenance,
        base_credits_spent=credit_cost,
        provider_credits_spent=round(provider_credits, 2),
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
