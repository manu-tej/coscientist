"""Contract tests against the real PubMed API. Opt in: pytest -m network

Offline fixtures agree with our parser by construction, so only a real call can
catch NCBI changing its response shape — or our queries being malformed.
"""
import httpx
import pytest

from tools.evidence import PubMedSource
from tools.pubmed import PubMedTool, _get, to_query

pytestmark = pytest.mark.network


async def _esearch_count(term: str) -> int:
    """Record count for a raw esearch term. Goes through `_get`, the same
    throttled/retried path production code uses, so probing here can't itself
    trip NCBI's rate limit."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await _get(client, "esearch.fcgi", {
            "db": "pubmed", "term": term, "retmax": 0, "retmode": "json",
        })
        return int(response.json()["esearchresult"]["count"])


async def test_returns_real_records_with_abstracts():
    outcome = await PubMedSource(max_results=3).search_outcome(
        "sildenafil pulmonary hypertension")
    assert outcome.failed_sources == []
    assert outcome.records
    assert all(r.ref.isdigit() for r in outcome.records)
    assert any(r.has_abstract for r in outcome.records), (
        "no abstracts returned — efetch is not working")


async def test_claim_shaped_query_returns_hits():
    """The production shape: to_query() must turn an unsearchable claim
    sentence into a searchable one. Self-asserting rather than resting on a
    docstring claim that can silently rot: fetches both counts live and
    asserts the actual contrast. If PubMed's index ever makes the raw sentence
    searchable, the premise this test guards is gone, and it says so rather
    than passing for the wrong reason.

    SMAD3-PAI1 promoter binding is real, established TGF-beta/fibrosis biology
    (PAI1/SERPINE1 is a canonical SMAD3 target gene), but this specific
    phrasing — SMAD3 + PAI1 + promoter + hepatic stellate cells, all
    co-occurring — has very little indexed literature (2 records at last
    check), which is exactly why it survives only through to_query's cleanup
    and not verbatim. A live search for a "SMAD3 binds X promoter" claim with
    a larger, more robust hit count was not found: every combination checked
    that returned comfortably more than a couple of hits also returned hits
    verbatim, and every verbatim-zero combination found stayed in the
    low single digits after to_query — see the task report for the sentences
    probed. That means this test's margin is thin by nature of the property
    it guards, not from a lack of trying; the live assertions below exist so
    that if literature growth (or removal) ever erodes that margin to zero,
    the test fails loudly with a diagnosis instead of quietly proving nothing.
    """
    claim = "SMAD3 binds the PAI1 promoter in hepatic stellate cells"
    verbatim_count = await _esearch_count(claim)
    reduced_count = await _esearch_count(to_query(claim))
    assert verbatim_count == 0, (
        "premise broken: sentence is searchable verbatim, so this no longer "
        "guards to_query")
    assert reduced_count > 0, "to_query failed to make a claim sentence searchable"

    # Also exercise the actual production path (esearch+esummary+efetch),
    # not just the raw esearch count above.
    articles = await PubMedTool(max_results=3).search(claim)
    assert articles, "claim-shaped query returned nothing; to_query is not working"


async def test_negation_does_not_invert_the_query():
    """Bare `not` is a PubMed boolean operator: the raw sentence returns 9391
    records that EXCLUDE sildenafil."""
    articles = await PubMedTool(max_results=3).search("sildenafil does not inhibit PDE3")
    assert any("sildenafil" in (a["title"] + a["content"]).lower() for a in articles)


async def test_very_long_input_does_not_414():
    articles = await PubMedTool(max_results=2).search("pulmonary hypertension " * 200)
    assert isinstance(articles, list)
