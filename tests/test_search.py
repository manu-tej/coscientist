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
    tool._get_client()  # force lazy construction so _client is not None before patching
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
    tool._get_client()  # force lazy construction so _client is not None before patching
    with patch.object(tool._client, "search", return_value=mock_results):
        formatted = await tool.search_and_format("ALS", "Article on ALS mechanisms")

    assert "Study A" in formatted
    assert "Content A" in formatted


async def test_search_tool_no_api_key_does_not_crash_on_construction():
    # Constructing with an empty key must NOT raise (entrypoint passes "" when
    # TAVILY_API_KEY is unset). The system can run with PubMed instead.
    tool = SearchTool(api_key="", max_results=3)
    assert tool is not None


async def test_search_no_key_returns_empty():
    tool = SearchTool(api_key="", max_results=3)
    articles = await tool.search("anything")
    assert articles == []


async def test_search_and_format_no_key_returns_fallback():
    tool = SearchTool(api_key="", max_results=3)
    formatted = await tool.search_and_format("anything")
    assert "No relevant articles found." in formatted
