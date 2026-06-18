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
    on tournament leaders. The gating lives here; wiring a real GPU provider is the
    next increment (Modal), for which a T0-priced placeholder already exercises it.
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
                value = a.run(inputs)
            except Exception:  # a real provider erroring must not sink the cascade
                value = None
            if value is not None:
                return ModelOutput(facet, a.name, a.tier, round(float(value), 4),
                                   a.credits, detail=f"{a.tier.value} via {a.name}")
        raise RegistryError(
            f"no provider produced a value for facet {facet!r} "
            f"(allow_expensive={allow_expensive}) — missing T0 floor?")


# --- default providers (CPU-real / offline) ---------------------------------

def _kg_proximity(inputs: dict) -> Optional[float]:
    return kg.proximity_score(inputs["drug"], inputs["indication"])


def _fixture_estimate(inputs: dict) -> float:
    """The T0 floor for a per-method vote: the candidate's pre-seeded fixture score
    stands in for an LLM/heuristic estimate. Always answers (never abstains)."""
    return float(inputs.get("fixture_score", 0.0))


def _t0_affinity_stub(inputs: dict) -> float:
    """Placeholder for Modal Boltz-2 binding affinity. Real provider is the next
    increment; today this returns a T0 estimate so the cascade gating is exercised."""
    return float(inputs.get("prior", 0.5))


def build_default_registry(*, enable_literature: bool = False,
                           literature_fetch=None) -> ModelRegistry:
    """The v1 registry. Facets:

    network_proximity — KG-on-CPU (T1) over a fixture T0 floor.
    literature        — PubMed co-occurrence (T1, network) over a fixture T0 floor,
                        wired ONLY when enable_literature (keeps default runs offline
                        and deterministic). `literature_fetch` is the injectable
                        count fn (for tests); abstains → T0 floor on any failure.
    affinity          — a (future) Modal GPU provider priced ~25cr to exercise the
                        cost-tiered cascade; today only the T0 estimate floor answers.
    """
    reg = ModelRegistry()
    reg.register(FnAdapter("kg_cpu", "network_proximity", FidelityTier.T1, _kg_proximity, credits=0.0))
    reg.register(FnAdapter("t0_fixture", "network_proximity", FidelityTier.T0, _fixture_estimate, credits=0.0))

    if enable_literature:
        from company import literature
        reg.register(FnAdapter(
            "pubmed", "literature", FidelityTier.T1,
            lambda i: literature.literature_score(i["drug"], i["indication"], fetch=literature_fetch),
            credits=0.0))
    reg.register(FnAdapter("t0_literature", "literature", FidelityTier.T0, _fixture_estimate, credits=0.0))

    reg.register(FnAdapter("modal_boltz2", "affinity", FidelityTier.T2, lambda i: None, credits=25.0))
    reg.register(FnAdapter("t0_affinity", "affinity", FidelityTier.T0, _t0_affinity_stub, credits=0.0))
    return reg
