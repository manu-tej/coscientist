"""KG-on-CPU generation method — network-medicine proximity (build-path step 1).

The FIRST live, CPU-real generation method wired into `science.run_stage`. It
replaces the hand-typed `network` per-method score on each candidate with one
*computed* from a small curated drug→target→disease knowledge graph, using a
real network-medicine proximity (best weight-product path, à la Guney et al.
2016). No API key, no network, no large download → deterministic and offline,
so it honours the v1 spine's reproducibility contract while being T1 (real CPU
computation) rather than T0 (LLM/heuristic estimate).

The graph is a small *curated seed* (real drug-target and gene-disease edges for
the PAH demo), NOT PrimeKG/DRKG. Swapping in a full KGE / TxGNN adapter is the
same seam: keep the `proximity_score(drug, indication) -> float | None` contract;
a method that can't speak to a candidate returns None (abstains) so it composes
with the other voting methods instead of forcing a spurious zero.
"""
from __future__ import annotations

import math
from functools import lru_cache

import networkx as nx

# --- curated seed graph (real edges, qualitative weights) -------------------
# Drug → (gene target, binding weight). Approved drugs with well-established
# molecular targets; weights are qualitative "we know it binds" confidences.
_DRUG_TARGETS: dict[str, list[tuple[str, float]]] = {
    "sildenafil":    [("PDE5A", 0.95)],
    "tadalafil":     [("PDE5A", 0.93)],
    "imatinib":      [("PDGFRB", 0.85), ("ABL1", 0.90), ("KIT", 0.85)],
    "metformin":     [("PRKAA1", 0.70)],
    "spironolactone": [("NR3C2", 0.90)],
    "loratadine":    [("HRH1", 0.95)],   # negative control — off-module target
}

# Gene → (disease, association weight). The PAH "disease module": genes with
# real evidence linking them to pulmonary arterial hypertension biology.
_PAH = "pulmonary arterial hypertension"
_GENE_DISEASE: dict[str, float] = {
    "PDE5A": 0.95,    # NO–cGMP vasodilation (sildenafil/tadalafil target)
    "EDNRA": 0.90, "EDNRB": 0.85,  # endothelin axis (bosentan/ambrisentan)
    "PTGIR": 0.90,    # prostacyclin receptor (epoprostenol/treprostinil)
    "GUCY1A1": 0.90,  # soluble guanylate cyclase (riociguat)
    "PDGFRB": 0.70,   # anti-remodeling (imatinib) — efficacy/tolerability tradeoff
    "BMPR2": 0.95,    # principal genetic driver (no approved drug)
    "PRKAA1": 0.45,   # AMPK — preclinical anti-proliferative signal (metformin)
    "NR3C2": 0.40,    # mineralocorticoid receptor — adjunctive interest only
}

# Protein–protein edges that give the network real depth (a drug whose target is
# one hop off the module can still reach the disease). Functional, not invented.
_PPI: list[tuple[str, str, float]] = [
    ("PDE5A", "GUCY1A1", 0.60),  # both in the NO–cGMP pathway
]


@lru_cache(maxsize=1)
def _graph() -> "nx.Graph":
    g = nx.Graph()
    for drug, targets in _DRUG_TARGETS.items():
        for gene, w in targets:
            g.add_edge(drug, gene, weight=w)
    for gene, w in _GENE_DISEASE.items():
        g.add_edge(gene, _PAH, weight=w)
    for a, b, w in _PPI:
        g.add_edge(a, b, weight=w)
    return g


def known_drugs() -> set[str]:
    return set(_DRUG_TARGETS)


def proximity_score(drug: str, indication: str) -> float | None:
    """Network-medicine proximity of `drug` to `indication` in [0, 1], or None if
    the method can't reach a verdict (drug/disease absent, or no connecting path).

    Score = the maximum product of edge weights along any path from drug to disease
    (computed as a min-cost shortest path under cost = -log weight). A direct
    drug→target→disease route scores binding×association; longer routes through PPI
    edges are multiplicatively discounted — exactly the distance penalty network
    medicine wants, here on a weighted graph so evidence strength shows through.
    """
    g = _graph()
    drug = drug.strip().lower()
    if drug not in g or indication.strip().lower() not in (_PAH,):
        return None
    try:
        dist = nx.shortest_path_length(g, drug, _PAH, weight=lambda u, v, d: -math.log(d["weight"]))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return round(math.exp(-dist), 4)


def annotate_network_scores(candidates: list) -> int:
    """Overwrite each candidate's `network` method score with the computed KG
    proximity (real, T1). A candidate the KG can't score abstains (score 0.0, which
    `integrate_candidate` then ignores). Returns how many got a live, non-None score.
    """
    scored = 0
    for c in candidates:
        s = proximity_score(c.drug, c.indication)
        c.method_scores["network"] = s if s is not None else 0.0
        if s is not None:
            scored += 1
    return scored
