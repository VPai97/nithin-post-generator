import os
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class ResearchResult:
    title: str
    url: str
    snippet: str


class ResearchClient:
    """Lightweight web research via external search APIs."""

    def __init__(self):
        self.provider = (os.environ.get("RESEARCH_PROVIDER") or "").strip().lower()
        self.api_key = (
            os.environ.get("RESEARCH_API_KEY")
            or os.environ.get("TAVILY_API_KEY")
            or os.environ.get("SERPER_API_KEY")
            or os.environ.get("BRAVE_API_KEY")
        )

    def is_available(self) -> bool:
        return bool(self.provider and self.api_key)

    def search(self, query: str, max_results: int = 5) -> list[ResearchResult]:
        if not self.is_available():
            return []

        if self.provider == "tavily":
            return self._search_tavily(query, max_results)
        if self.provider == "serper":
            return self._search_serper(query, max_results)
        if self.provider == "brave":
            return self._search_brave(query, max_results)

        return []

    def _search_tavily(self, query: str, max_results: int) -> list[ResearchResult]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results
        }
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("results", []):
            results.append(
                ResearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", "") or item.get("snippet", "")
                )
            )
        return results

    def _search_serper(self, query: str, max_results: int) -> list[ResearchResult]:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": self.api_key}
        payload = {"q": query, "num": max_results}
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("organic", []):
            results.append(
                ResearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", "")
                )
            )
        return results

    def _search_brave(self, query: str, max_results: int) -> list[ResearchResult]:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"X-Subscription-Token": self.api_key}
        params = {"q": query, "count": max_results}
        response = requests.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("web", {}).get("results", []):
            results.append(
                ResearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", "")
                )
            )
        return results
