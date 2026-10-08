from typing import Protocol

from pydantic import BaseModel, Field, HttpUrl


class WebSearchResult(BaseModel):
    title: str = Field(min_length=1)
    url: HttpUrl
    snippet: str = Field(min_length=1)


class WebSearch(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int = 3,
    ) -> list[WebSearchResult]:
        ...