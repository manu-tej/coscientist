"""Uniform interface over literature sources.

Sources report failure via SearchOutcome.failed_sources instead of swallowing it.
An adversarial review found that swallowing made "the source was down" and "the
literature has nothing" indistinguishable — which turned outages into recorded
scientific findings.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceRecord:
    source: str
    ref: str
    title: str
    snippet: str
    url: str
    year: int | None = None
    has_abstract: bool = False


@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    status: str          # supported | unsupported | contradicted | unchecked
    records: list[EvidenceRecord] = field(default_factory=list)
    rationale: str = ""


@dataclass(frozen=True)
class SearchOutcome:
    records: list[EvidenceRecord] = field(default_factory=list)
    failed_sources: list[str] = field(default_factory=list)


VALID_STATUSES = ("supported", "unsupported", "contradicted", "unchecked")


@runtime_checkable
class EvidenceSource(Protocol):
    name: str

    async def search_outcome(self, query: str, *, limit: int = 5) -> SearchOutcome:
        ...


def _year_from(*texts: str) -> int | None:
    for text in texts:
        match = re.search(r"\b(19|20)\d{2}\b", text or "")
        if match:
            return int(match.group(0))
    return None


def _pmid_from(url: str) -> str:
    match = re.search(r"/(\d+)/?$", url or "")
    return match.group(1) if match else (url or "")


class PubMedSource:
    name = "pubmed"

    def __init__(self, client=None, max_results: int = 5, api_key: str | None = None):
        if client is None:
            from tools.pubmed import PubMedTool
            client = PubMedTool(max_results=max_results, api_key=api_key)
        self.client = client

    async def search_outcome(self, query: str, *, limit: int = 5) -> SearchOutcome:
        try:
            articles = await self.client.search(query)
            records = [
                EvidenceRecord(
                    source=self.name,
                    ref=_pmid_from(a.get("url", "")),
                    title=a.get("title", ""),
                    snippet=a.get("content", ""),
                    url=a.get("url", ""),
                    year=_year_from(a.get("pubdate", ""), a.get("content", "")),
                    has_abstract=bool(a.get("has_abstract")),
                )
                for a in articles[:limit]
            ]
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            # Covers both the client raising (PubMedTool's documented failure
            # mode) and the client returning a malformed payload (e.g. a
            # non-dict element or a non-sliceable object) — this layer must
            # not rely on the client's shape guarantee to stay non-raising.
            logger.warning("pubmed search failed (%s)", exc)
            return SearchOutcome(records=[], failed_sources=[self.name])
        return SearchOutcome(records=records)

    async def search(self, query: str, *, limit: int = 5) -> list[EvidenceRecord]:
        """Plain-list convenience wrapper. Discards `failed_sources` — callers
        that need failure visibility must use `search_outcome()` instead."""
        return (await self.search_outcome(query, limit=limit)).records


class WebSource:
    """Adapter so the Tavily SearchTool satisfies EvidenceSource.

    Public (no leading underscore) because coscientist.py imports it.
    """

    name = "web"

    def __init__(self, client):
        self.client = client

    async def search_outcome(self, query: str, *, limit: int = 5) -> SearchOutcome:
        # The mapping stays INSIDE the try: a malformed article dict must be
        # reported as a failed source, not raised out of the layer whose whole
        # job is converting exceptions into reported state.
        try:
            articles = await self.client.search(query)
            records = []
            for a in articles[:limit]:
                url = a.get("url", "") or ""
                # A PubMed URL returned via web search must key on the SAME
                # bare PMID PubMedSource would use, or set_grounding_refs and
                # fabricated_citations compare apples to oranges: a model that
                # cites a record it was shown (by PMID) reads as fabrication,
                # and the LiteratureTool dedup below fails to merge the two
                # copies of the same paper. Gated on hostname rather than a
                # bare _pmid_from(url), which would turn any trailing-digit
                # non-PubMed URL into a fake PMID collision.
                ref = _pmid_from(url) if "pubmed.ncbi.nlm.nih.gov" in url else url
                records.append(EvidenceRecord(
                    source=self.name, ref=ref, title=a.get("title", ""),
                    snippet=(a.get("content", "") or "")[:1200], url=url,
                    year=_year_from(a.get("content", "")), has_abstract=False,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("web search failed (%s)", exc)
            return SearchOutcome(records=[], failed_sources=[self.name])
        return SearchOutcome(records=records)


class LiteratureTool:
    """Composite over N sources. Drop-in for the historical SearchTool."""

    name = "literature"

    def __init__(self, sources: list[EvidenceSource], max_results: int = 5):
        self.sources = list(sources)
        self.max_results = max_results

    async def search_outcome(self, query: str, *, limit: int | None = None) -> SearchOutcome:
        limit = limit or self.max_results
        # Each source gets the full limit; dedup then truncation yields `limit`
        # merged records rather than `limit // len(sources)` from each.
        results = await asyncio.gather(
            *(s.search_outcome(query, limit=limit) for s in self.sources),
            return_exceptions=True,
        )
        merged, seen, failed = [], set(), []
        for source, result in zip(self.sources, results):
            if isinstance(result, BaseException):
                logger.warning("evidence source %s raised (%s)", source.name, result)
                failed.append(source.name)
                continue
            failed.extend(result.failed_sources)
            for record in result.records:
                if record.ref in seen:
                    continue
                seen.add(record.ref)
                merged.append(record)
        return SearchOutcome(records=merged[:limit], failed_sources=failed)

    async def search(self, query: str, *, limit: int | None = None) -> list[EvidenceRecord]:
        return (await self.search_outcome(query, limit=limit)).records

    async def search_and_format(self, query: str, context: str = "") -> str:
        outcome = await self.search_outcome(query)
        if not outcome.records:
            logger.warning(
                "no evidence for %r (failed sources: %s); this hypothesis will be "
                "ungrounded", query[:80], outcome.failed_sources or "none",
            )
            return "No relevant articles found."
        lines = []
        for i, r in enumerate(outcome.records, 1):
            lines.append(f"[{i}] {r.title} ({r.source.upper()}:{r.ref})")
            lines.append(f"URL: {r.url}")
            lines.append(f"Summary: {r.snippet[:1200]}")
            lines.append("")
        return "\n".join(lines)


class FixtureEvidenceSource:
    """Deterministic offline evidence. Synthetic, and labelled as such.

    Refs are deliberately non-numeric so a fixture citation can never be mistaken
    for a real PMID. has_abstract is True so the offline path can exercise the
    contradiction rule, which requires abstract-backed evidence.
    """

    name = "fixture"

    def __init__(self, seed: int = 0):
        self.seed = seed

    async def search_outcome(self, query: str, *, limit: int = 5) -> SearchOutcome:
        import hashlib
        digest = int.from_bytes(
            hashlib.sha256(f"{self.seed}\x00{query}".encode()).digest()[:4], "big")
        count = 1 + digest % 3
        return SearchOutcome(records=[
            EvidenceRecord(
                source=self.name,
                ref=f"OFFLINE-{(digest + i) % 100000:05d}",
                title=f"Offline fixture record {i + 1} (synthetic, not a real paper)",
                snippet=("Deterministic offline evidence for pipeline testing. "
                         "Carries no scientific meaning."),
                url="https://example.invalid/offline-fixture",
                year=2000 + ((digest + i) % 25),
                has_abstract=True,
            )
            for i in range(min(count, limit))
        ])

    async def search(self, query: str, *, limit: int = 5) -> list[EvidenceRecord]:
        return (await self.search_outcome(query, limit=limit)).records


def build_sources(provider: str, max_results: int, tavily_key: str = "") -> list:
    """Shared factory so coscientist.py and bench/runner.py wire identically."""
    if provider == "offline":
        # Defined in Task 7; imported lazily so Task 3 stands alone.
        from tools.evidence import FixtureEvidenceSource
        return [FixtureEvidenceSource()]
    sources: list = [PubMedSource(max_results=max_results)]
    if tavily_key:
        from tools.search import SearchTool
        sources.append(WebSource(SearchTool(api_key=tavily_key, max_results=max_results)))
    return sources
