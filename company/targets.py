"""Target protein resolution — the second input structure-based methods need.

Structure/affinity methods need a protein sequence, not a disease name. This maps
a disease to its principal druggable target (UniProt accession) and pulls the
sequence from the UniProt REST API, cached on disk.

Crucially it slices the **ligand-binding domain** rather than the full chain:
Boltz-2 cost scales steeply with sequence length, and the ligand binds the
catalytic pocket, so docking the full protein pays for residues that don't
participate. The domain is chosen by annotation — the one containing the most
annotated active/binding sites — not by a hardcoded guess.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_CACHE = Path(".cache/targets")
_UNIPROT = "https://rest.uniprot.org/uniprotkb"


@dataclass
class Target:
    gene: str
    accession: str
    disease: str


# Principal druggable target per disease — each is the actual target of that
# disease's known actives (PDE5A for the PDE5 inhibitors, CRBN for the IMiDs,
# ESR1 for the SERMs, SRD5A2 for finasteride).
TARGETS: dict[str, Target] = {
    "pulmonary arterial hypertension": Target("PDE5A", "O76074", "pulmonary arterial hypertension"),
    "erectile dysfunction": Target("PDE5A", "O76074", "erectile dysfunction"),
    "multiple myeloma": Target("CRBN", "Q96SW2", "multiple myeloma"),
    "breast cancer": Target("ESR1", "P03372", "breast cancer"),
    "androgenetic alopecia": Target("SRD5A2", "P31213", "androgenetic alopecia"),
}


def target_for(disease: str) -> Optional[Target]:
    return TARGETS.get(disease.strip().lower())


def cache_key(region_label: str) -> str:
    """Stable MSA cache key from a region label, e.g. 'O76074 binding domain (536-860)'
    -> 'O76074_536-860'. Keyed on accession AND range so the cached alignment is
    unambiguous about which construct it describes."""
    acc = region_label.split()[0]
    rng = region_label.split("(")[-1].rstrip(")") if "(" in region_label else "full"
    return f"{acc}_{rng}"


def _fetch_json(accession: str) -> dict:
    _CACHE.mkdir(parents=True, exist_ok=True)
    cached = _CACHE / f"{accession}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    import httpx
    r = httpx.get(f"{_UNIPROT}/{accession}.json", timeout=30, follow_redirects=True)
    r.raise_for_status()
    data = r.json()
    cached.write_text(json.dumps(data))
    return data


def _binding_domain(features: list) -> Optional[tuple[int, int]]:
    """The annotated domain containing the most active/binding sites — i.e. the
    pocket the ligand actually occupies. None if no domain qualifies."""
    def span(f):
        loc = f["location"]
        return loc["start"].get("value"), loc["end"].get("value")

    domains = [span(f) for f in features if f["type"] == "Domain"]
    domains = [(s, e) for s, e in domains if s and e]
    sites = [span(f)[0] for f in features
             if f["type"] in ("Active site", "Binding site") and span(f)[0]]
    if not domains:
        return None
    if not sites:
        return max(domains, key=lambda d: d[1] - d[0])
    best = max(domains, key=lambda d: sum(1 for s in sites if d[0] <= s <= d[1]))
    covered = sum(1 for s in sites if best[0] <= s <= best[1])
    return best if covered else None


def sequence_for(disease: str, *, full_chain: bool = False) -> Optional[tuple[str, str, str]]:
    """(gene, sequence, region_label) for a disease's target, or None if unmapped.

    Returns the ligand-binding domain slice unless `full_chain`. The region label
    records exactly what was docked, so the evidence graph can state it.
    """
    target = target_for(disease)
    if target is None:
        return None
    try:
        data = _fetch_json(target.accession)
    except Exception:
        return None
    seq = data.get("sequence", {}).get("value")
    if not seq:
        return None
    if full_chain:
        return target.gene, seq, f"{target.accession} full chain (1-{len(seq)})"
    dom = _binding_domain(data.get("features", []))
    if dom is None:
        return target.gene, seq, f"{target.accession} full chain (1-{len(seq)})"
    start, end = dom
    return (target.gene, seq[start - 1:end],
            f"{target.accession} binding domain ({start}-{end})")
