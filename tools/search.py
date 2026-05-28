from tavily import TavilyClient


class SearchTool:
    def __init__(self, api_key: str, max_results: int = 5):
        self._client = TavilyClient(api_key=api_key)
        self.max_results = max_results

    async def search(self, query: str) -> list[dict]:
        response = self._client.search(
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
