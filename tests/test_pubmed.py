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
        resp.json = MagicMock(return_value=esearch_json)
        resp.raise_for_status = MagicMock()
        return resp

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.get = AsyncMock(side_effect=fake_get)
        articles = await tool.search("nonexistent topic xyz")

    assert articles == []
