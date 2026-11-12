"""Tests for tools/evidence.py (the literature/verification layer).

Named test_tools_evidence to avoid colliding with tests/test_evidence.py, which
covers company/evidence.py's provenance graph.
"""
import pytest

from tools.evidence import EvidenceRecord, PubMedSource, SearchOutcome, WebSource


class _FakePubMed:
    def __init__(self, articles):
        self.articles = articles

    async def search(self, query):
        return self.articles


async def test_maps_articles_to_records_with_abstract_flag():
    source = PubMedSource(client=_FakePubMed([{
        "title": "PDE5 in stellate cells",
        "url": "https://pubmed.ncbi.nlm.nih.gov/28442/",
        "content": "PKG-I phosphorylates RhoA at Ser188.",
        "journal": "J Hepatol", "pubdate": "2017 Mar", "has_abstract": True,
    }]))

    outcome = await source.search_outcome("PDE5 fibrosis")

    assert outcome.failed_sources == []
    record = outcome.records[0]
    assert record.ref == "28442"
    assert record.has_abstract is True
    assert record.year == 2017


async def test_transport_failure_is_reported_not_swallowed():
    """The whole point: a failing source must be distinguishable from an empty one."""
    class _Boom:
        async def search(self, query):
            raise RuntimeError("NCBI unreachable")

    outcome = await PubMedSource(client=_Boom()).search_outcome("anything")

    assert outcome.records == []
    assert outcome.failed_sources == ["pubmed"]


async def test_empty_result_is_not_a_failure():
    outcome = await PubMedSource(client=_FakePubMed([])).search_outcome("anything")
    assert outcome.records == []
    assert outcome.failed_sources == []


async def test_search_keeps_the_plain_list_interface():
    source = PubMedSource(client=_FakePubMed([]))
    assert await source.search("q") == []


async def test_malformed_success_payload_is_reported_not_raised():
    """The client can succeed (no exception) yet still hand back a shape that
    breaks the record-mapping code — e.g. non-dict elements. That must be
    reported via failed_sources, not propagate out of search_outcome."""
    outcome = await PubMedSource(client=_FakePubMed(["not a dict"])).search_outcome("anything")

    assert outcome.records == []
    assert outcome.failed_sources == ["pubmed"]


class _FakeWebClient:
    def __init__(self, articles):
        self.articles = articles

    async def search(self, query):
        return self.articles


# CRITICAL 1: WebSource used to key every record on the raw URL, even for
# PubMed hits returned via web search. PubMedSource keys on the bare PMID.
# That mismatch broke both set_grounding_refs/fabricated_citations comparison
# and LiteratureTool's ref-based dedup.


async def test_web_source_pubmed_url_yields_bare_pmid_ref():
    source = WebSource(_FakeWebClient([{
        "title": "PDE5 in stellate cells",
        "url": "https://pubmed.ncbi.nlm.nih.gov/28442/",
        "content": "PKG-I phosphorylates RhoA at Ser188.",
    }]))
    outcome = await source.search_outcome("PDE5 fibrosis")
    assert outcome.records[0].ref == "28442"


async def test_web_source_non_pubmed_url_keeps_full_url_as_ref():
    source = WebSource(_FakeWebClient([{
        "title": "A blog post",
        "url": "https://example.com/posts/28442",
        "content": "Not PubMed.",
    }]))
    outcome = await source.search_outcome("q")
    assert outcome.records[0].ref == "https://example.com/posts/28442"


async def test_web_source_malformed_success_payload_is_reported_not_raised():
    """WebSource is on the critical path (Critical 1 lived here) -- the
    client can succeed yet hand back a shape that breaks record-mapping."""
    source = WebSource(_FakeWebClient(["not a dict"]))
    outcome = await source.search_outcome("anything")
    assert outcome.records == []
    assert outcome.failed_sources == ["web"]


from tools.evidence import LiteratureTool


class _Stub:
    def __init__(self, name, records, fails=False):
        self.name, self._records, self._fails = name, records, fails

    async def search_outcome(self, query, *, limit=5):
        from tools.evidence import SearchOutcome
        if self._fails:
            return SearchOutcome(records=[], failed_sources=[self.name])
        return SearchOutcome(records=self._records[:limit])


def _rec(ref, source="pubmed"):
    return EvidenceRecord(source=source, ref=ref, title=f"T{ref}", snippet="s",
                          url=f"https://x/{ref}", year=2020, has_abstract=True)


async def test_merges_and_dedupes_by_ref():
    tool = LiteratureTool(sources=[_Stub("pubmed", [_rec("1"), _rec("2")]),
                                   _Stub("web", [_rec("2"), _rec("3")])])
    outcome = await tool.search_outcome("q")
    assert [r.ref for r in outcome.records] == ["1", "2", "3"]


async def test_failed_source_is_propagated_not_hidden():
    tool = LiteratureTool(sources=[_Stub("pubmed", [], fails=True),
                                   _Stub("web", [_rec("9")])])
    outcome = await tool.search_outcome("q")
    assert outcome.failed_sources == ["pubmed"]
    assert [r.ref for r in outcome.records] == ["9"]


async def test_total_failure_reports_every_source():
    tool = LiteratureTool(sources=[_Stub("pubmed", [], fails=True),
                                   _Stub("web", [], fails=True)])
    outcome = await tool.search_outcome("q")
    assert sorted(outcome.failed_sources) == ["pubmed", "web"]


async def test_empty_result_is_logged_not_silent(caplog):
    tool = LiteratureTool(sources=[_Stub("pubmed", [])])
    with caplog.at_level("WARNING"):
        text = await tool.search_and_format("q")
    assert "No relevant articles found." in text
    assert any("no evidence" in r.message.lower() for r in caplog.records)


async def test_dedups_same_paper_returned_by_pubmed_and_web():
    """CRITICAL 1: before the ref-keying fix, PubMedSource keyed on the bare
    PMID while WebSource keyed on the full URL, so the same paper surfaced
    from both sources produced two records and was presented to the judge as
    two independent [N] sources instead of being merged into one."""
    same_paper = {
        "title": "PDE5 in stellate cells",
        "url": "https://pubmed.ncbi.nlm.nih.gov/28442/",
        "content": "PKG-I phosphorylates RhoA at Ser188.",
        "pubdate": "2017", "has_abstract": True,
    }
    tool = LiteratureTool(sources=[
        PubMedSource(client=_FakePubMed([same_paper])),
        WebSource(_FakeWebClient([same_paper])),
    ])
    outcome = await tool.search_outcome("PDE5 fibrosis")
    assert len(outcome.records) == 1
    assert outcome.records[0].ref == "28442"


async def test_format_includes_citable_refs():
    tool = LiteratureTool(sources=[_Stub("pubmed", [_rec("28442")])])
    assert "28442" in await tool.search_and_format("q")
