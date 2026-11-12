# tests/test_pubmed.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tools.pubmed import PubMedTool


@pytest.fixture
def tool():
    return PubMedTool(max_results=2)


async def test_search_returns_articles(tool):
    esearch_json = {"esearchresult": {"idlist": ["111", "222"]}}
    esummary_json = {
        "result": {
            "uids": ["111", "222"],
            "111": {"title": "ALS Study", "fulljournalname": "Nature", "pubdate": "2024"},
            "222": {"title": "Motor Neuron Paper", "fulljournalname": "Cell", "pubdate": "2023"},
        }
    }

    async def fake_get(url, params=None):
        resp = MagicMock()
        resp.status_code = 200
        if "esearch" in url:
            resp.json = MagicMock(return_value=esearch_json)
        else:
            resp.json = MagicMock(return_value=esummary_json)
        resp.raise_for_status = MagicMock()
        return resp

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.get = AsyncMock(side_effect=fake_get)
        articles = await tool.search("ALS mechanisms")

    assert len(articles) == 2
    assert articles[0]["title"] == "ALS Study"
    assert "Nature" in articles[0]["content"]


async def test_search_and_format(tool):
    esearch_json = {"esearchresult": {"idlist": ["111"]}}
    esummary_json = {
        "result": {
            "uids": ["111"],
            "111": {"title": "ALS Study", "fulljournalname": "Nature", "pubdate": "2024"},
        }
    }

    async def fake_get(url, params=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.json = MagicMock(return_value=esearch_json if "esearch" in url else esummary_json)
        resp.raise_for_status = MagicMock()
        return resp

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.get = AsyncMock(side_effect=fake_get)
        formatted = await tool.search_and_format("ALS")

    assert "ALS Study" in formatted
    assert "[1]" in formatted


async def test_search_handles_empty_results(tool):
    esearch_json = {"esearchresult": {"idlist": []}}

    async def fake_get(url, params=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.json = MagicMock(return_value=esearch_json)
        resp.raise_for_status = MagicMock()
        return resp

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.get = AsyncMock(side_effect=fake_get)
        articles = await tool.search("nonexistent topic xyz")

    assert articles == []


# --- Task 1 additions: throttle, abstracts, query construction ---
import asyncio
import time

from tools.pubmed import PubMedTool, to_query


def test_to_query_strips_prose_into_searchable_terms():
    q = to_query("SMAD3 binds the PDE5A promoter")
    assert "SMAD3" in q and "PDE5A" in q
    assert " the " not in f" {q} "          # stopwords gone
    assert "binds" not in q                  # verbs gone


def test_to_query_neutralises_boolean_operators():
    """PubMed parses bare not/and/or as operators. 'sildenafil does not inhibit
    PDE3' returns 9391 records that EXCLUDE sildenafil — worse than zero."""
    q = to_query("sildenafil does not inhibit PDE3")
    assert " not " not in f" {q} "
    assert " NOT " not in f" {q} "
    assert "sildenafil" in q.lower()


def test_to_query_caps_length():
    """A markdown task doc as a goal produced HTTP 414 against the real API.

    Regression: the original probe ("word " * 500) dedups to a single term
    before to_query ever reaches the char cap, so the cap was never actually
    exercised. Unique long terms are needed so the built query genuinely
    exceeds max_chars and the trailing slice does real work.
    """
    text = " ".join(f"gene{i:03d}" + "x" * 30 for i in range(30))
    q = to_query(text)
    assert len(q) == 380
    assert len(q) < len(text)


def test_to_query_keeps_short_keyword_queries_intact():
    assert to_query("sildenafil pulmonary hypertension").lower().count("sildenafil") == 1


async def test_search_returns_abstract_as_content(monkeypatch):
    """esummary carries no abstract; without efetch the judge sees only a title."""
    calls = []

    class _Resp:
        def __init__(self, payload, text=""):
            self._payload, self.text, self.status_code = payload, text, 200

        def json(self):
            return self._payload

        def raise_for_status(self):
            pass

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            calls.append(url)
            if "esearch" in url:
                return _Resp({"esearchresult": {"idlist": ["111"]}})
            if "esummary" in url:
                return _Resp({"result": {"111": {
                    "title": "A study", "fulljournalname": "J Test", "pubdate": "2017 Mar",
                }}})
            return _Resp({}, text=(
                "<PubmedArticle><PMID>111</PMID>"
                "<AbstractText>PKG-I phosphorylates RhoA at Ser188.</AbstractText>"
                "</PubmedArticle>"
            ))

    monkeypatch.setattr("tools.pubmed.httpx.AsyncClient", lambda *a, **k: _Client())
    articles = await PubMedTool(max_results=1).search("PKG-I RhoA")

    assert any("efetch" in c for c in calls), "must fetch abstracts, not just summaries"
    assert articles[0]["content"] == "PKG-I phosphorylates RhoA at Ser188."
    assert articles[0]["has_abstract"] is True
    assert articles[0]["journal"] == "J Test"


async def test_throttle_paces_concurrent_calls():
    """8 concurrent requests at defaults reproduced live 429s from NCBI."""
    from tools.pubmed import _throttled

    stamps = []

    async def _fake():
        async with _throttled():
            stamps.append(time.monotonic())

    await asyncio.gather(*(_fake() for _ in range(4)))
    gaps = [b - a for a, b in zip(sorted(stamps), sorted(stamps)[1:])]
    assert all(g >= 0.30 for g in gaps), f"calls not paced: {gaps}"


async def test_search_raises_when_retries_exhausted(monkeypatch):
    """The whole point of the throttle/retry rewrite: a persistent 429 must
    RAISE so a caller can record `unchecked`, never silently return [] --
    which the layer above would read as 'no such literature exists'."""
    import httpx as httpx_mod

    calls = []

    async def _fast_sleep(*a, **k):
        return None

    # Only the retry backoff / throttle pacing sleeps are stubbed out; the
    # retry bound (3 attempts, see tools/pubmed.py::_get) is untouched.
    monkeypatch.setattr("tools.pubmed.asyncio.sleep", _fast_sleep)

    class _Resp:
        status_code = 429
        headers = {}

        def raise_for_status(self):
            raise httpx_mod.HTTPStatusError("429 rate limited", request=None, response=self)

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            calls.append(url)
            return _Resp()

    monkeypatch.setattr("tools.pubmed.httpx.AsyncClient", lambda *a, **k: _Client())

    with pytest.raises(httpx_mod.HTTPError):
        await PubMedTool(max_results=1).search("PKG-I RhoA")

    # Bounded: exactly the 3 attempts in _get's `for attempt in range(3)`, not
    # an infinite retry loop.
    assert len(calls) == 3, f"expected 3 bounded attempts, got {len(calls)}: {calls}"


async def test_search_degrades_to_metadata_when_efetch_persistently_fails(monkeypatch):
    """esearch/esummary succeed but efetch never does (persistent 5xx). search()
    must still return the records -- with has_abstract=False -- rather than
    failing the whole call, per Task 1 resolution #5."""
    import httpx as httpx_mod

    calls = []

    async def _fast_sleep(*a, **k):
        return None

    monkeypatch.setattr("tools.pubmed.asyncio.sleep", _fast_sleep)

    class _Resp:
        def __init__(self, payload=None, status_code=200):
            self._payload = payload
            self.status_code = status_code
            self.headers = {}

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx_mod.HTTPStatusError("error", request=None, response=self)

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            calls.append(url)
            if "esearch" in url:
                return _Resp({"esearchresult": {"idlist": ["111"]}})
            if "esummary" in url:
                return _Resp({"result": {"111": {
                    "title": "A study", "fulljournalname": "J Test", "pubdate": "2017 Mar",
                }}})
            return _Resp(status_code=503)   # efetch: persistently unavailable

    monkeypatch.setattr("tools.pubmed.httpx.AsyncClient", lambda *a, **k: _Client())

    articles = await PubMedTool(max_results=1).search("PKG-I RhoA")

    assert len(articles) == 1
    assert articles[0]["title"] == "A study"
    assert articles[0]["journal"] == "J Test"
    assert articles[0]["has_abstract"] is False
    assert articles[0]["content"] == "J Test (2017 Mar)"

    efetch_calls = [c for c in calls if "efetch" in c]
    assert len(efetch_calls) == 3, f"expected 3 bounded efetch attempts, got {len(efetch_calls)}"


def test_query_ladder_broadens_progressively():
    """A real extracted claim ANDs into zero hits. Live run afc8d1be (2026-08-10)
    returned 0 records for 8/8 claims about well-published biology, and every one
    was recorded as `unsupported` — documented to users as evidence of absence.
    The ladder is what makes that verdict honest."""
    from tools.pubmed import query_ladder

    claim = ("Phosphorylation of EGFR at Thr693 located in the kinase activation "
             "segment reduces the receptor intrinsic tyrosine kinase activity")
    rungs = query_ladder(claim)

    assert len(rungs) >= 3, "ladder must offer progressively broader attempts"
    # Strictly broadening: each rung ANDs no more terms than the one before it.
    ands = [r.count(" AND ") for r in rungs]
    assert ands == sorted(ands, reverse=True), f"rungs not broadening: {ands}"
    # The last rung must not be a pure conjunction of many terms.
    assert " OR " in rungs[-1] or rungs[-1].count(" AND ") <= 1


async def test_search_falls_back_when_the_narrow_query_finds_nothing(monkeypatch):
    """The first rung returning zero must not end the search."""
    from tools.pubmed import PubMedTool

    attempts = []

    class _Resp:
        def __init__(self, payload, text=""):
            self._p, self.text, self.status_code = payload, text, 200
        def json(self): return self._p
        def raise_for_status(self): pass

    class _Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None):
            if "esearch" in url:
                attempts.append(params["term"])
                # Only the broadest attempt finds anything.
                ids = ["777"] if len(attempts) >= 2 else []
                return _Resp({"esearchresult": {"idlist": ids}})
            if "esummary" in url:
                return _Resp({"result": {"777": {
                    "title": "Found via fallback", "fulljournalname": "J", "pubdate": "2020",
                }}})
            return _Resp({}, text="<PubmedArticle><PMID>777</PMID>"
                                  "<AbstractText>Real abstract.</AbstractText></PubmedArticle>")

    monkeypatch.setattr("tools.pubmed.httpx.AsyncClient", lambda *a, **k: _Client())
    articles = await PubMedTool(max_results=3).search(
        "Uroguanylin encoded by GUCA2B is severely downregulated in colorectal adenomas")

    assert len(attempts) >= 2, "search gave up after the first empty rung"
    assert articles and articles[0]["title"] == "Found via fallback"
    assert articles[0]["has_abstract"] is True


def test_query_ladder_prefers_identifiers_over_long_common_words():
    """Gene symbols and residue identifiers are SHORT; ranking by length drops
    exactly the discriminative terms. Live run 063ba003 (2026-08-10) retrieved a
    herpesvirus paper for a claim about MYPT1/Thr696/ROCK because the ladder kept
    'phosphorylates' and 'cytoskeletal' and discarded the gene symbols."""
    from tools.pubmed import query_ladder

    claim = ("PKG I phosphorylates MYPT1 at Thr696 to suppress ROCK kinase "
             "activity toward cytoskeletal substrates")
    narrowest = query_ladder(claim)[0]

    assert "MYPT1" in narrowest, f"dropped the gene symbol: {narrowest}"
    assert "Thr696" in narrowest, f"dropped the residue identifier: {narrowest}"
    assert "ROCK" in narrowest, f"dropped the kinase: {narrowest}"
    assert "cytoskeletal" not in narrowest, (
        f"kept a long generic word over an identifier: {narrowest}")


def test_query_ladder_scores_identifier_shapes():
    from tools.pubmed import _specificity

    # digits (residues, isoforms) and all-caps symbols beat long generic words
    assert _specificity("Thr696") > _specificity("phosphorylates")
    assert _specificity("PDE5A") > _specificity("preferentially")
    assert _specificity("SMAD3") > _specificity("transcription")
    assert _specificity("ROCK") > _specificity("substrates")
