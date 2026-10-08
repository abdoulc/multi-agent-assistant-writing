import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from .web_search import WebSearchResult


class TavilySearch:
    def __init__(self, api_key: str, *, timeout: float = 30):
        if not api_key.strip():
            raise ValueError("A Tavily API key is required.")
        if timeout <= 0:
            raise ValueError("Timeout must be positive.")
        self.api_key = api_key.strip()
        self.timeout = timeout

    def search(
        self, query: str, *, limit: int = 3
    ) -> list[WebSearchResult]:
        if not query.strip():
            raise ValueError("Search query must not be blank.")
        if not 1 <= limit <= 20:
            raise ValueError("Search limit must be between 1 and 20.")

        payload = {
            "query": query.strip(),
            "max_results": limit,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }
        request = Request(
            "https://api.tavily.com/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"Web search HTTP error: {exc.code}.") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError("Web search connection failed or timed out.") from exc
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("Web search returned invalid JSON.") from exc

        try:
            items = data["results"]
            if not isinstance(items, list):
                raise TypeError("Expected a results list.")
            return [
                WebSearchResult(
                    title=item["title"], url=item["url"], snippet=item["content"]
                )
                for item in items[:limit]
            ]
        except (KeyError, TypeError, ValidationError) as exc:
            raise RuntimeError("Web search returned an invalid response.") from exc