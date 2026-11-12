"""PubMed E-utilities client.

Three things here are load-bearing and were added after an adversarial review:

1. A module-level throttle. NCBI allows ~3 req/s unkeyed; one reflection task at
   default settings otherwise fires 8 concurrent requests and gets 429s, which
   the layer above would record as "the literature says no".

2. efetch for abstracts. esummary has NO abstract field, so without this the
   claim judge decides from a title and a journal name.

3. Query construction. PubMed ANDs every content word, so a claim sentence
   returns 0 hits, and a bare `not` is parsed as a boolean operator.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import re
import time
# defusedxml, not stdlib ElementTree: efetch returns third-party XML, and the
# stdlib parser is vulnerable to XXE and billion-laughs by default.
from defusedxml import ElementTree as ET

import httpx

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# 0.35s matches company/literature.py:25, which paces the same API for the same
# reason. Two concurrent slots keeps burst rate under the unkeyed 3 req/s ceiling.
# Module-level, so it assumes one event loop per process (asyncio.Semaphore and
# asyncio.Lock bind to the loop that first awaits them) -- true for every entry
# point in this codebase today (coscientist.py, bench.runner, company.hierarchy
# all run a single asyncio.run()). Reviewed and accepted as-is: a multi-loop
# process would need a per-loop gate, but that is a re-architecture, not a fix.
NCBI_MIN_INTERVAL = 0.35
_NCBI_GATE = asyncio.Semaphore(2)
_pace_lock = asyncio.Lock()
_last_call = 0.0

_STOPWORDS = frozenset("""
a an the of in on at to for with by from as is are was were be been being
this that these those it its their his her our your my we you they he she
and or not but if then than when while which who whom whose what how why
does do did done can could may might must should would will shall
via using use used induces induce induced binds bind bound causes cause caused
increases increase decreased decreases elevates elevate leads lead may involve
""".split())


@contextlib.asynccontextmanager
async def _throttled():
    """Serialise and pace outbound NCBI requests process-wide."""
    global _last_call
    async with _NCBI_GATE:
        async with _pace_lock:
            wait = NCBI_MIN_INTERVAL - (time.monotonic() - _last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            _last_call = time.monotonic()
        yield


def to_query(text: str, *, max_terms: int = 12, max_chars: int = 380) -> str:
    """Turn prose into a PubMed-safe term query.

    PubMed ANDs every content word, so a full sentence usually returns nothing;
    and boolean operator words must never survive verbatim.
    """
    cleaned = re.sub(r"[^\w\s-]", " ", text or "")
    terms, seen = [], set()
    for raw in cleaned.split():
        word = raw.strip("-")
        low = word.lower()
        if not word or low in _STOPWORDS or len(word) < 3:
            continue
        if low in seen:
            continue
        seen.add(low)
        terms.append(word)
        if len(terms) >= max_terms:
            break
    query = " AND ".join(terms) if terms else (text or "").strip()[:max_chars]
    return query[:max_chars]


def _specificity(term: str) -> int:
    """How much a token pins a claim to particular papers.

    Biomedical identifiers are short and high-signal; generic scientific verbs
    and adjectives are long and low-signal. Length is therefore an actively
    misleading proxy — this scores identifier *shape* instead.
    """
    score = 0
    if any(ch.isdigit() for ch in term):
        score += 3                      # Thr696, PDE5A, LATS1, G12D
    if term.isupper() and len(term) >= 3:
        score += 2                      # ROCK, CDK8, SMAD3, IPF
    elif term[:1].isupper() and any(ch.isupper() for ch in term[1:]):
        score += 2                      # MyPT1-style mixed case
    if "-" in term:
        score += 1                      # N-myristoylated, IL-13
    if len(term) <= 6 and term.islower():
        score -= 1                      # short generic words: among, most
    return score


def query_ladder(text: str) -> list[str]:
    """Progressively broader queries to try until one returns records.

    `to_query` alone is not enough: it ANDs every surviving term, so a real
    extracted claim ("Phosphorylation of EGFR at Thr693 located in the kinase
    activation segment...") becomes a 12-way conjunction that matches nothing.
    Live run afc8d1be returned 0 records for 8/8 claims about well-published
    biology, and the layer above recorded each as `unsupported` — which the
    README defines to users as "the literature was searched and had nothing to
    say". A false negative here is exactly the overstatement this system exists
    to avoid, so we must exhaust reasonable phrasings before reporting absence.

    Ordering is narrow-to-broad so the most precise query that works wins.
    """
    terms = [t for t in to_query(text, max_terms=12).split(" AND ") if t]
    if not terms:
        return []
    # Rank by how *identifying* a token is, not by how long it is. Gene symbols
    # and residue identifiers are short (MYPT1, ROCK, Thr696, PDE5A) while the
    # least discriminative words are long (phosphorylates, cytoskeletal). Ranking
    # by length dropped exactly the terms that pin a claim to its literature and
    # retrieved a herpesvirus paper for a MYPT1/Thr696/ROCK claim (run 063ba003).
    ranked = sorted(range(len(terms)),
                    key=lambda i: (-_specificity(terms[i]), i))

    def top(n: int) -> list[str]:
        keep = sorted(ranked[:n])
        return [terms[i] for i in keep]

    rungs: list[str] = []
    for n in (6, 4, 3, 2):
        if len(terms) >= n:
            rungs.append(" AND ".join(top(n)))
    if not rungs:
        rungs.append(" AND ".join(terms))
    # Final rung: any of the most distinctive terms, so a real but
    # differently-phrased literature base is still found.
    rungs.append(" OR ".join(top(min(4, len(terms)))))
    # De-duplicate while preserving order.
    seen, unique = set(), []
    for rung in rungs:
        if rung not in seen:
            seen.add(rung)
            unique.append(rung)
    return unique


async def _get(client, endpoint: str, params: dict):
    """One throttled GET, honouring Retry-After, then raising so callers can
    map the failure to `unchecked` rather than to a false negative result."""
    for attempt in range(3):
        async with _throttled():
            response = await client.get(f"{_EUTILS}/{endpoint}", params=params)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 2:
                response.raise_for_status()
            await asyncio.sleep(float(response.headers.get("Retry-After", 1)) + attempt)
            continue
        response.raise_for_status()
        return response
    raise httpx.HTTPError("NCBI retries exhausted")


def _abstracts_from(xml_text: str) -> dict[str, str]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:      # ParseError, EntitiesForbidden, DTDForbidden
        return {}
    out = {}
    for article in root.iter("PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None:
            continue
        parts = [(el.text or "").strip() for el in article.iter("AbstractText")]
        if parts:
            out[(pmid_el.text or "").strip()] = " ".join(p for p in parts if p)
    return out


class PubMedTool:
    def __init__(self, max_results: int = 5, api_key: str | None = None):
        self.max_results = max_results
        # A free NCBI key raises the ceiling from 3 to 10 req/s.
        self.api_key = api_key or os.environ.get("NCBI_API_KEY") or None

    def _auth(self, params: dict) -> dict:
        if self.api_key:
            params = {**params, "api_key": self.api_key}
        return params

    async def search(self, query: str) -> list[dict]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Walk narrow-to-broad and stop at the first rung with hits. Reporting
            # "no matching records" off a single over-conjoined query produces
            # false negatives that the layer above records as evidence of absence.
            ids: list[str] = []
            for term in query_ladder(query) or [to_query(query)]:
                r = await _get(client, "esearch.fcgi", self._auth({
                    "db": "pubmed", "term": term, "retmax": self.max_results,
                    "retmode": "json", "sort": "relevance",
                }))
                ids = r.json().get("esearchresult", {}).get("idlist", [])
                if ids:
                    break
            if not ids:
                return []

            r2 = await _get(client, "esummary.fcgi", self._auth({
                "db": "pubmed", "id": ",".join(ids), "retmode": "json",
            }))
            summaries = r2.json().get("result", {})

            # One batched efetch regardless of PMID count.
            try:
                r3 = await _get(client, "efetch.fcgi", self._auth({
                    "db": "pubmed", "id": ",".join(ids),
                    "rettype": "abstract", "retmode": "xml",
                }))
                abstracts = _abstracts_from(r3.text)
            except httpx.HTTPError:
                abstracts = {}   # degrade to metadata; callers see has_abstract=False

        articles = []
        for pmid in ids:
            meta = summaries.get(pmid, {})
            if not meta:
                continue
            journal = meta.get("fulljournalname", "")
            pubdate = meta.get("pubdate", "")
            abstract = abstracts.get(pmid, "")
            articles.append({
                "title": meta.get("title", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "content": abstract or f"{journal} ({pubdate})",
                "journal": journal,
                "pubdate": pubdate,
                "has_abstract": bool(abstract),
            })
        return articles

    async def search_and_format(self, query: str, context: str = "") -> str:
        articles = await self.search(query)
        if not articles:
            return "No relevant articles found."
        lines = []
        for i, a in enumerate(articles, 1):
            lines.append(f"[{i}] {a['title']}")
            lines.append(f"URL: {a['url']}")
            lines.append(f"Summary: {a['content'][:1200]}")
            lines.append("")
        return "\n".join(lines)
