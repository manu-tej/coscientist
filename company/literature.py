"""Literature co-occurrence — a second, independent live generation method (spec §4).

Where the KG method (company/kg.py) scores a (drug → indication) pair from graph
structure, this scores it from the *text record*: how often do the drug and the
disease co-occur in PubMed? Two independent data sources → their agreement is a
real multi-method consensus signal, not the same evidence counted twice.

Unlike the KG method this one is network-bound and non-deterministic, so it is
T1 (real, but not reproducible) and is **opt-in**: the registry only wires it
when explicitly enabled, and it *abstains* (returns None) on any error or offline
so resolution falls through to the T0 fixture floor — tests stay deterministic and
offline by default. The `fetch` seam makes the normalization unit-testable without
touching the network.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_CAP = 1000          # co-occurrence count at which the score saturates to ~1.0
_TIMEOUT = 10.0


def _ncbi_count(term: str, *, api_key: Optional[str] = None) -> int:
    """Hit NCBI E-utilities esearch and read the total hit count (keyless, rate-limited)."""
    import httpx
    params = {"db": "pubmed", "term": term, "rettype": "count", "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    r = httpx.get(f"{_EUTILS}/esearch.fcgi", params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return int(r.json()["esearchresult"]["count"])


def cooccurrence_count(drug: str, disease: str, *,
                       fetch: Optional[Callable[[str], int]] = None) -> Optional[int]:
    """PubMed co-occurrence count for the (drug, disease) pair, or None on failure.

    `fetch(term) -> int` is injectable for tests; defaults to the live NCBI query.
    """
    fetch = fetch or _ncbi_count
    term = f'"{drug}"[tiab] AND "{disease}"[tiab]'
    try:
        return int(fetch(term))
    except Exception:   # network down, rate-limited, parse error — abstain, don't crash
        return None


def literature_score(drug: str, disease: str, *,
                     fetch: Optional[Callable[[str], int]] = None) -> Optional[float]:
    """Co-occurrence count mapped to [0, 1] by log saturation, or None if unavailable.

    log1p scaling so a handful of papers already lifts the score off the floor while
    thousands saturate near 1 — established repurposings (sildenafil↔PAH) score high,
    spurious pairs score low.
    """
    n = cooccurrence_count(drug, disease, fetch=fetch)
    if n is None:
        return None
    return round(min(1.0, math.log1p(n) / math.log1p(_CAP)), 4)
