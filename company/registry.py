"""Model registry + adapters — the seam every scientific method plugs into (spec §5).

A *facet* (e.g. "network_proximity", "affinity") maps to an ordered list of
**providers** of decreasing fidelity, ending in a T0 estimate floor so resolution
never fails. `resolve(facet, inputs)` walks the list and returns the first provider
that produces a value, recording which **fidelity tier** answered — so the gate
packet can always say whether a number was really computed or estimated.

Two design properties from the spec:
  • graceful fallback — real provider abstains (returns None) → next provider tried →
    T0 floor always answers. Mirrors `tools/llm.py`'s factory.
  • cost-tiered cascade — expensive providers (Modal GPU) are gated behind
    `allow_expensive`; cheap CPU methods screen everything, the pricey ones run only
    on portfolio leaders. The registry includes real optional Modal structure and
    affinity adapters, with T0 fallbacks when those providers are disabled or
    abstain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol, runtime_checkable

from company import kg


class FidelityTier(str, Enum):
    T0 = "T0"   # LLM/heuristic estimate — the always-available floor
    T1 = "T1"   # real CPU computation (e.g. KG proximity, RDKit)
    T2 = "T2"   # real SOTA model (Modal GPU / hosted API)


class RegistryError(RuntimeError):
    """No provider produced a value for a facet — a misconfigured registry (no T0 floor)."""


@dataclass
class ModelOutput:
    facet: str
    provider: str
    tier: FidelityTier
    value: float
    credits: float = 0.0
    detail: str = ""
    evidence: dict = field(default_factory=dict)   # concrete provenance (path, citation, …)


@runtime_checkable
class ModelAdapter(Protocol):
    name: str
    facet: str
    tier: FidelityTier
    credits: float

    def run(self, inputs: dict) -> Optional[float]:
        """Return a value in the facet's range, or None to abstain (fall through)."""
        ...


@dataclass
class FnAdapter:
    """A provider backed by a plain function `(inputs) -> float | None`."""
    name: str
    facet: str
    tier: FidelityTier
    fn: Callable[[dict], Optional[float]]
    credits: float = 0.0

    def run(self, inputs: dict) -> Optional[float]:
        return self.fn(inputs)


class ModelRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, list[ModelAdapter]] = {}

    def register(self, adapter: ModelAdapter) -> None:
        self._providers.setdefault(adapter.facet, []).append(adapter)

    def providers(self, facet: str) -> list[ModelAdapter]:
        return list(self._providers.get(facet, []))

    def max_provider_credits(
        self,
        facet: str,
        *,
        allow_expensive: bool = True,
        expensive_threshold: float = 10.0,
    ) -> float:
        """Conservative pre-call reservation for one facet resolution."""
        eligible = [
            float(adapter.credits)
            for adapter in self.providers(facet)
            if allow_expensive or adapter.credits <= expensive_threshold
        ]
        return max(eligible, default=0.0)

    def resolve(self, facet: str, inputs: dict, *, allow_expensive: bool = True,
                expensive_threshold: float = 10.0) -> ModelOutput:
        """First provider (in registration order) that yields a non-None value wins.

        Providers costing more than `expensive_threshold` credits are skipped unless
        `allow_expensive` (the cost-tiered cascade: run GPU only on leaders). A facet
        with a T0 floor always resolves; otherwise RegistryError (fail loud).
        """
        for a in self.providers(facet):
            if a.credits > expensive_threshold and not allow_expensive:
                continue
            try:
                result = a.run(inputs)
            except Exception:  # a real provider erroring must not sink the cascade
                result = None
            if result is None:
                continue
            # A provider may return a bare value, or (value, evidence) to carry its
            # concrete provenance (KG path, citation count) into the evidence graph.
            value, ev = result if isinstance(result, tuple) else (result, {})
            return ModelOutput(facet, a.name, a.tier, round(float(value), 4),
                               a.credits, detail=f"{a.tier.value} via {a.name}",
                               evidence=ev or {})
        raise RegistryError(
            f"no provider produced a value for facet {facet!r} "
            f"(allow_expensive={allow_expensive}) — missing T0 floor?")


# --- default providers (CPU-real / offline) ---------------------------------

def _kg_proximity(inputs: dict):
    """Returns (score, evidence) so the mechanism path reaches the evidence graph."""
    ev = kg.proximity_evidence(inputs["drug"], inputs["indication"])
    if ev is None:
        return None
    score, path = ev
    return score, {"kind": "kg_path", "path": path}


def _fixture_estimate(inputs: dict) -> float:
    """The T0 floor for a per-method vote: the candidate's pre-seeded fixture score
    stands in for an LLM/heuristic estimate. Always answers (never abstains)."""
    return float(inputs.get("fixture_score", 0.0))


# ChemBERTa cosines are compressed: unrelated molecules still score ~0.45-0.65,
# so the raw value would inflate the averaged confidence. Rescale against the
# observed unrelated-molecule floor. PROVISIONAL — anchored on a handful of pairs;
# recalibrate on a larger sample before treating the magnitude as meaningful.
_CHEM_FLOOR = 0.45


def _rescale_cosine(cos: float) -> float:
    return max(0.0, min(1.0, (cos - _CHEM_FLOOR) / (1.0 - _CHEM_FLOOR)))


def _modal_structure(inputs: dict):
    """T2 structure method: chemical analogy to the disease's OTHER known actives.

    Leave-one-out (see company.chem.reference_actives) keeps this a real prediction
    rather than a lookup. Abstains — falling through to the T0 floor — whenever the
    drug has no SMILES, there are no references, or Modal is unreachable/unauthed.
    """
    from company import chem

    query = chem.smiles_for(inputs["drug"])
    refs = chem.reference_smiles(inputs["drug"], inputs["indication"])
    if not query or not refs:
        return None
    try:
        import modal
        embedder = modal.Cls.from_name("coscientist-chem", "ChemEmbedder")()
        res = embedder.max_similarity.remote(query, [s for _, s in refs])
    except Exception:      # modal missing, unauthenticated, network down — abstain
        return None
    if not res or res.get("score") is None:
        return None
    cos = float(res["score"])
    nearest = refs[int(res["nearest_index"])][0]
    return _rescale_cosine(cos), {"kind": "analog", "nearest": nearest, "cosine": round(cos, 4)}


# Sigmoid centre, as log10(IC50 / µM). -1.0 == 100 nM: the potency an APPROVED drug
# typically needs against its target to matter clinically at achievable exposure.
# Assumes MSA-mode predictions. Single-sequence output runs ~2 log units weak (see
# _potency_score), so scoring it against this centre understates everything.
_POTENCY_CENTRE = -1.0


def _potency_score(pred_value: float) -> float:
    """Boltz-2 affinity regression → [0,1] potency score.

    `affinity_pred_value` is predicted log10(IC50) in µM, LOWER meaning tighter
    binding. Mapped through a Hill-shaped sigmoid centred on the drug-potency
    threshold, monotonically decreasing in predicted IC50.

    Two findings, both measured on the PAH pool vs PDE5A, drove this:

    1. Use the regression head, NOT `affinity_probability_binary`. The binary head
       ranked the loratadine negative control above tadalafil and put spironolactone
       first, with every value in a mushy 0.19-0.55 band. It screens binders from
       decoys; it is not a potency rank.

    2. The centre must match the input mode. With a cached MSA, sildenafil predicts
       6.5 nM against a true ~3.5 nM — accurate. Single-sequence predicted 285 nM,
       ~80x weak. An earlier 10 µM centre appeared to work only because it was
       compensating for that systematic bias; on accurate MSA values it saturates
       (top-4 spread 0.034 vs 0.723 here). Fix the input, then set the centre on
       pharmacology rather than tuning it to rescue a broken one.
    """
    return round(1.0 / (1.0 + 10.0 ** (pred_value - _POTENCY_CENTRE)), 4)


def _modal_boltz2(inputs: dict):
    """T2 affinity: Boltz-2 protein-ligand binding on Modal A100.

    Needs BOTH a ligand SMILES and a target binding-domain sequence; abstains to
    the T0 floor if either is unresolved or Modal is unreachable. Boltz-2's binary
    affinity regression value is converted to a pharmacology-anchored potency
    score. This is the provider priced above the cascade threshold, so it runs
    only for portfolio leaders.
    """
    from company import chem, targets

    smiles = chem.smiles_for(inputs["drug"])
    if not smiles:
        return None  # checked first: resolving the target fetches from UniProt
    resolved = targets.sequence_for(inputs["indication"])
    if not resolved:
        return None
    gene, sequence, region = resolved
    try:
        import modal
        # Ensure the target's MSA exists (computed once on CPU, then cached in the
        # volume — returns instantly on every subsequent ligand for this target).
        # Without it Boltz-2 runs single-sequence and predicts ~2 log units weak.
        key = targets.cache_key(region)
        msa_fn = modal.Function.from_name("coscientist-chem", "compute_target_msa")
        msa = msa_fn.remote(sequence, key)
        msa_path = msa.get("path", "") if msa.get("ok") else ""

        fn = modal.Function.from_name("coscientist-chem", "predict_affinity")
        res = fn.remote(sequence, smiles, name=f"{inputs['drug'][:12]}", msa_path=msa_path)
    except Exception:
        return None
    if not res or not res.get("ok") or res.get("affinity_pred_value") is None:
        return None
    pred = float(res["affinity_pred_value"])
    return _potency_score(pred), {
        "kind": "affinity", "target": gene, "region": region,
        "pred_value": round(pred, 3), "elapsed_s": res.get("elapsed_s"),
        "predicted_nM": round(10 ** pred * 1000, 1), "msa": bool(msa_path),
    }


def _t0_affinity_stub(inputs: dict) -> float:
    """Offline affinity floor used when the optional Modal Boltz-2 provider abstains."""
    return float(inputs.get("prior", 0.5))


def build_default_registry(*, enable_literature: bool = False,
                           literature_fetch=None,
                           enable_structure: bool = False,
                           enable_affinity: bool = False) -> ModelRegistry:
    """The v1 registry. Facets:

    network_proximity — KG-on-CPU (T1) over a fixture T0 floor.
    literature        — PubMed co-occurrence (T1, network) over a fixture T0 floor,
                        wired ONLY when enable_literature (keeps default runs offline
                        and deterministic). `literature_fetch` is the injectable
                        count fn (for tests); abstains → T0 floor on any failure.
    affinity          — optional Modal Boltz-2 T2 provider over a T0 estimate
                        floor, gated by the cost-tiered cascade.
    """
    reg = ModelRegistry()
    reg.register(FnAdapter("kg_cpu", "network_proximity", FidelityTier.T1, _kg_proximity, credits=0.0))
    reg.register(FnAdapter("t0_fixture", "network_proximity", FidelityTier.T0, _fixture_estimate, credits=0.0))

    if enable_literature:
        from company import literature

        def _pubmed(i: dict):
            ev = literature.literature_evidence(i["drug"], i["indication"], fetch=literature_fetch)
            if ev is None:
                return None
            score, count = ev
            return score, {"kind": "citation", "count": count,
                           "query": literature.query_term(i["drug"], i["indication"])}

        reg.register(FnAdapter("pubmed", "literature", FidelityTier.T1, _pubmed, credits=0.0))
    reg.register(FnAdapter("t0_literature", "literature", FidelityTier.T0, _fixture_estimate, credits=0.0))

    # structure: real Modal GPU (ChemBERTa) analogy scoring, opt-in. Priced honestly
    # low — a T4 embedding call IS cheap, so it screens broadly rather than being
    # gated. Boltz-2 is the one that will sit above the cascade threshold.
    if enable_structure:
        reg.register(FnAdapter("modal_chemberta", "structure", FidelityTier.T2,
                               _modal_structure, credits=2.0))
    reg.register(FnAdapter("t0_structure", "structure", FidelityTier.T0, _fixture_estimate, credits=0.0))

    # affinity: real Boltz-2 on A100, opt-in AND priced above the cascade threshold
    # so it only runs for portfolio leaders (allow_expensive) — the expensive-GPU
    # gating the cascade was built for.
    if enable_affinity:
        reg.register(FnAdapter("modal_boltz2", "affinity", FidelityTier.T2,
                               _modal_boltz2, credits=25.0))
    reg.register(FnAdapter("t0_affinity", "affinity", FidelityTier.T0, _t0_affinity_stub, credits=0.0))
    return reg
