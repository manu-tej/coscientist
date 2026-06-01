import httpx

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedTool:
    def __init__(self, max_results: int = 5, api_key: str | None = None):
        self.max_results = max_results
        self.api_key = api_key

    async def search(self, query: str) -> list[dict]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            esearch_params = {
                "db": "pubmed",
                "term": query,
                "retmax": self.max_results,
                "retmode": "json",
                "sort": "relevance",
            }
            if self.api_key:
                esearch_params["api_key"] = self.api_key
            r = await client.get(f"{_EUTILS}/esearch.fcgi", params=esearch_params)
            r.raise_for_status()
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            esummary_params = {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "json",
            }
            if self.api_key:
                esummary_params["api_key"] = self.api_key
            r2 = await client.get(f"{_EUTILS}/esummary.fcgi", params=esummary_params)
            r2.raise_for_status()
            result = r2.json().get("result", {})

            articles = []
            for pmid in ids:
                meta = result.get(pmid, {})
                if not meta:
                    continue
                title = meta.get("title", "")
                journal = meta.get("fulljournalname", "")
                pubdate = meta.get("pubdate", "")
                articles.append({
                    "title": title,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "content": f"{journal} ({pubdate})",
                })
            return articles

    async def search_and_format(self, query: str, context: str = "") -> str:
        articles = await self.search(query)
        if not articles:
            return "No relevant articles found."
        lines = []
        for i, a in enumerate(articles, 1):
            lines.append(f"[{i}] {a['title']}")
            lines.append(f"URL: {a['url']}")
            lines.append(f"Summary: {a['content']}")
            lines.append("")
        return "\n".join(lines)
