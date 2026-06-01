from tavily import TavilyClient


class SearchTool:
    def __init__(self, api_key: str, max_results: int = 5):
        self.api_key = api_key
        self.max_results = max_results
        self._client = None  # lazily constructed; None until a keyed search runs

    def _get_client(self):
        if self._client is None and self.api_key:
            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    async def search(self, query: str) -> list[dict]:
        client = self._get_client()
        if client is None:
            # No API key configured — degrade gracefully (use PubMed instead).
            return []
        response = client.search(
            query=query,
            max_results=self.max_results,
            search_depth="advanced",
        )
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in response.get("results", [])
        ]

    async def search_and_format(self, query: str, context: str = "") -> str:
        articles = await self.search(query)
        if not articles:
            return "No relevant articles found."
        lines = []
        for i, a in enumerate(articles, 1):
            lines.append(f"[{i}] {a['title']}")
            lines.append(f"URL: {a['url']}")
            lines.append(f"Summary: {a['content'][:500]}")
            lines.append("")
        return "\n".join(lines)
