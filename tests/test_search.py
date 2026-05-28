import pytest
from unittest.mock import patch, MagicMock
from tools.search import SearchTool


@pytest.fixture
def tool():
    return SearchTool(api_key="test-key", max_results=3)


async def test_search_returns_articles(tool):
    mock_results = {
        "results": [
            {"title": "ALS Study", "url": "http://example.com/1", "content": "ALS content"},
            {"title": "Motor Neuron", "url": "http://example.com/2", "content": "Motor content"},
        ]
    }
    with patch.object(tool._client, "search", return_value=mock_results):
        articles = await tool.search("ALS mechanisms")

    assert len(articles) == 2
    assert articles[0]["title"] == "ALS Study"
    assert "url" in articles[0]
    assert "content" in articles[0]


async def test_search_formats_for_prompt(tool):
    mock_results = {
        "results": [
            {"title": "Study A", "url": "http://a.com", "content": "Content A"},
        ]
    }
    with patch.object(tool._client, "search", return_value=mock_results):
        formatted = await tool.search_and_format("ALS", "Article on ALS mechanisms")

    assert "Study A" in formatted
    assert "Content A" in formatted
