"""Claim extraction and verification.

Failure honesty is the point of this module. "Could not check" (source down,
judge unparseable, evidence too weak) is `unchecked`; "checked and found nothing"
is `unsupported`. Merging them turns an outage into a scientific finding.
"""
from __future__ import annotations

import asyncio
import logging
import re

from agents.base import LABEL_DECO
from tools.evidence import ClaimVerdict, EvidenceRecord
from tools.resources import resource_path

logger = logging.getLogger(__name__)

_PROMPTS = resource_path("prompts")
_VERDICT_RE = re.compile(
    rf"^\s*{LABEL_DECO}verdict{LABEL_DECO}:{LABEL_DECO}"
    r"(supported|contradicted|unsupported)\b",
    re.IGNORECASE | re.MULTILINE,
)
_REASON_RE = re.compile(rf"^\s*{LABEL_DECO}reason{LABEL_DECO}:{LABEL_DECO}(.+)$",
                        re.IGNORECASE | re.MULTILINE)
# Strip retrieved-text lines that could be mistaken for the judge's own
# VERDICT/REASON output. Built from the same LABEL_DECO grammar as _VERDICT_RE
# so the two can never drift apart -- a decorated line that survives
# sanitisation but still matches the parser would be a live injection hole.
_INJECTION_RE = re.compile(rf"(?im)^{LABEL_DECO}(verdict|reason){LABEL_DECO}:.*$")
# Retrieved text must not be able to close the <evidence> fence early and get
# read back as prompt instead of data.
_FENCE_RE = re.compile(r"[<>]")


def _render(name: str, **variables) -> str:
    return (_PROMPTS / "verify" / f"{name}.txt").read_text().format(**variables)


def _parse_numbered(text: str, max_claims: int) -> list[str]:
    claims = []
    for line in (text or "").splitlines():
        # 8-char floor drops list scaffolding like "1. a" without dropping claims.
        match = re.match(r"\s*\d+[.)]\s+(.{8,})", line)
        if match:
            claims.append(match.group(1).strip().strip("*_").strip())
    return claims[:max_claims]


async def extract_claims(text: str, backend, *, max_claims: int = 4) -> list[str]:
    try:
        response = await backend.call(
            "You are a meticulous scientific fact-checker.",
            _render("extract_claims", text=text, max_claims=max_claims),
        )
        # A non-str response (backend misbehaving) must fall into the except
        # below rather than raise out of this function.
        return _parse_numbered(response, max_claims)
    except Exception as exc:  # noqa: BLE001
        logger.warning("claim extraction failed (%s); skipping verification", exc)
        return []


def _sanitise(text: str, limit: int = 1200) -> str:
    stripped = _INJECTION_RE.sub("", text or "")
    return _FENCE_RE.sub(" ", stripped)[:limit]


def _format_evidence(records: list[EvidenceRecord]) -> str:
    # IMPORTANT 4: `ref` was interpolated unsanitised -- title and snippet were
    # the only fields run through _sanitise. A ref is source-controlled data
    # (WebSource sets it from the retrieved URL/PMID) exactly like title and
    # snippet, so it can just as easily carry a crafted `</evidence>` fence
    # break or a decorated VERDICT/REASON line. Same defect class as the
    # CRITICAL injection issue this module already guards title/snippet
    # against.
    return "\n".join(
        f"[{i}] {_sanitise(r.title, 300)} ({r.source.upper()}:{_sanitise(r.ref, 200)}, "
        f"{r.year or 'n.d.'})"
        f"{'' if r.has_abstract else ' -- metadata only, no abstract'}\n"
        f"{_sanitise(r.snippet)}"
        for i, r in enumerate(records, 1)
    )


async def verify_claim(claim: str, sources, backend) -> ClaimVerdict:
    try:
        outcome = await sources.search_outcome(claim)
    except Exception as exc:  # noqa: BLE001
        return ClaimVerdict(claim=claim, status="unchecked",
                            rationale=f"evidence lookup raised: {exc}")

    if outcome.failed_sources and not outcome.records:
        # Could not check. Never report this as evidence of absence.
        return ClaimVerdict(claim=claim, status="unchecked",
                            rationale=f"sources unavailable: {', '.join(outcome.failed_sources)}")
    if not outcome.records:
        return ClaimVerdict(claim=claim, status="unsupported",
                            rationale="searched, no matching records found")

    records = outcome.records
    try:
        response = await backend.call(
            "You are a meticulous scientific fact-checker.",
            _render("judge_claim", claim=claim, evidence=_format_evidence(records)),
            use_strong=True,   # this verdict can veto the strong model's review
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("claim judgement failed for %r (%s)", claim[:60], exc)
        return ClaimVerdict(claim=claim, status="unchecked", records=records,
                            rationale=f"judgement call failed: {exc}")

    # A non-str response (backend misbehaving) must not reach findall/search.
    response = response if isinstance(response, str) else ""

    matches = _VERDICT_RE.findall(response)
    if not matches:
        logger.warning("judge returned no parseable verdict for %r", claim[:60])
        return ClaimVerdict(claim=claim, status="unchecked", records=records,
                            rationale="judge returned no parseable verdict")
    # Last match wins: defeats a preamble quote (e.g. an echoed record's text
    # winning before the real verdict). It does NOT defend a postamble -- a
    # quote appearing after the genuine verdict line would win instead. That
    # is an accepted tradeoff, not a guarantee against every ordering.
    status = matches[-1].lower()
    reason = _REASON_RE.search(response)
    rationale = (reason.group(1).strip().strip("*_") if reason else "")

    # Gate on the record the judge actually cited, not "any record has an
    # abstract" -- with mixed PubMed+Web evidence (the production default,
    # since WebSource always sets has_abstract=False), an incidental
    # abstract-backed record could otherwise launder a contradiction sourced
    # entirely from a metadata-only web snippet. An unparseable or
    # out-of-range citation errs toward unchecked, never toward a veto.
    cited = re.search(r"\[(\d+)\]", rationale)
    idx = int(cited.group(1)) - 1 if cited else None
    backed = idx is not None and 0 <= idx < len(records) and records[idx].has_abstract
    if status == "contradicted" and not backed:
        # A title and a journal name cannot justify an irreversible veto.
        return ClaimVerdict(claim=claim, status="unchecked", records=records,
                            rationale=f"metadata-only evidence; not judged: {rationale}")

    if outcome.failed_sources:
        # Some sources were unavailable even though others returned records
        # that got judged. The verdict is real, but its provenance is partial
        # -- downstream treats this rationale as provenance, so that must not
        # be silently indistinguishable from a run where every source worked.
        rationale = f"{rationale} [partial: {', '.join(outcome.failed_sources)} unavailable]"
    return ClaimVerdict(claim=claim, status=status, records=records, rationale=rationale)


class ClaimVerifier:
    """Extraction plus bounded-concurrency per-claim verification."""

    def __init__(self, max_claims: int = 4, concurrency: int = 2):
        self.max_claims = max_claims
        self.concurrency = concurrency

    async def verify(self, hypothesis_text: str, sources, backend) -> list[ClaimVerdict]:
        claims = await extract_claims(hypothesis_text, backend, max_claims=self.max_claims)
        if not claims:
            return []
        gate = asyncio.Semaphore(self.concurrency)

        async def _one(claim):
            async with gate:
                return await verify_claim(claim, sources, backend)

        # return_exceptions=True: one claim's unexpected failure (verify_claim
        # itself never raises by contract, but a bug or a Semaphore/asyncio
        # surprise should not sink every other claim's verdict) is converted
        # to an unchecked row instead of aborting the whole batch.
        results = await asyncio.gather(*(_one(c) for c in claims), return_exceptions=True)
        verdicts = []
        for claim, result in zip(claims, results):
            if isinstance(result, ClaimVerdict):
                verdicts.append(result)
            elif isinstance(result, asyncio.CancelledError):
                # Cancellation is control flow, not a claim-level failure --
                # let it propagate so a caller relying on task cancellation
                # (e.g. a timeout) still observes it.
                raise result
            else:
                verdicts.append(ClaimVerdict(claim=claim, status="unchecked",
                                             rationale=f"verification raised: {result}"))
        return verdicts
