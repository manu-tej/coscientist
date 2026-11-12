# tests/test_citation_check.py
from core.reflection_pipeline import fabricated_citations
from tools.evidence import WebSource


def test_detects_a_pmid_never_shown():
    assert fabricated_citations(
        "Supported by PMID:99999999 and PMID:28442.", shown={"28442"},
    ) == ["99999999"]


def test_no_citations_is_not_a_fabrication():
    assert fabricated_citations("No citations here.", shown={"28442"}) == []


def test_ignores_case_and_formatting():
    assert fabricated_citations("(pmid 28442) and PUBMED:28442", shown={"28442"}) == []


def test_unknown_grounding_set_cannot_accuse():
    """If we never recorded what was shown, we cannot call anything fabricated."""
    assert fabricated_citations("PMID:99999999", shown=None) == []


async def test_citation_to_a_web_sourced_pubmed_record_is_not_fabrication():
    """CRITICAL 1: WebSource used to key a PubMed hit on its full URL instead
    of the bare PMID that set_grounding_refs stores and fabricated_citations
    compares against. That mismatch made a model that cited a record it WAS
    shown (via web search) read as fabrication -- in a PubMed-outage case,
    turning an outage into a recorded false-fabrication finding."""
    class _Client:
        async def search(self, query):
            return [{
                "title": "PDE5 in stellate cells",
                "url": "https://pubmed.ncbi.nlm.nih.gov/28442/",
                "content": "PKG-I phosphorylates RhoA at Ser188.",
            }]

    outcome = await WebSource(_Client()).search_outcome("q")
    shown = {r.ref for r in outcome.records}

    assert shown == {"28442"}
    assert fabricated_citations("Supported (PMID: 28442).", shown=shown) == []
